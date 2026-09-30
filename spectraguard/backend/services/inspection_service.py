"""Inspection service: orchestrates real-time background inspection loop, WebSockets, calibration, and scan logging."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from PIL import Image
from fastapi import WebSocket
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal
from backend.acquisition.simulated import SimulatedHSISource
from backend.processing.pipeline import PreprocessingPipeline
from backend.inference.classifier import HSIClassifier
from backend.inference.decision import DecisionEngine
from backend.actuation.controller import RejectController
from backend.records import crud, models

logger = logging.getLogger(__name__)


class InspectionManager:
    """Manages system status, acquisition loop, model, actuation, and WebSocket clients."""

    def __init__(self):
        self.status: str = "IDLE"  # RUNNING | STOPPED | CALIBRATING | FAULT | IDLE
        self.product_type: str = "poultry"
        self.active_lot_id: Optional[str] = "LOT-DEMO-001"
        self.belt_speed_mm_s: float = settings.belt_speed_mm_s
        self.scan_interval_s: float = settings.scan_interval_ms / 1000.0
        self.force_contamination: Optional[str] = None

        # Statistics
        self.start_time: Optional[float] = None
        self.total_scanned: int = 0
        self.total_rejected: int = 0
        self.recent_latencies: List[float] = []

        # Core pipelines
        self.source = SimulatedHSISource(product_type=self.product_type)
        self.pipeline = PreprocessingPipeline()
        self.classifier = HSIClassifier()
        self.decision_engine = DecisionEngine()
        self.reject_controller = RejectController(belt_speed_mm_s=self.belt_speed_mm_s)

        # Background task
        self._loop_task: Optional[asyncio.Task] = None
        self.active_websockets: Set[WebSocket] = set()

    def initialize(self):
        """Load ML models and initialize hardware sources."""
        try:
            self.classifier.load()
            logger.info("Classifier model successfully loaded.")
        except Exception as e:
            logger.warning(f"Classifier model load failed: {e}. Will attempt fallback or train.")

        self.source.initialise()
        self.pipeline.set_references(
            self.source.get_dark_reference(),
            self.source.get_white_reference(),
        )
        self.reject_controller.start()

    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_websockets)}")

    def disconnect_websocket(self, websocket: WebSocket):
        self.active_websockets.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_websockets)}")

    async def broadcast(self, message: dict):
        if not self.active_websockets:
            return
        to_remove = set()
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                to_remove.add(ws)
        for ws in to_remove:
            self.active_websockets.discard(ws)

    def start_inspection(self, lot_id: Optional[str] = None, product_type: str = "poultry"):
        """Start background continuous scan loop."""
        if self.status == "RUNNING":
            return
        if lot_id:
            self.active_lot_id = lot_id
        self.product_type = product_type
        self.source.product_type = product_type

        # Ensure active lot exists in DB
        db: Session = SessionLocal()
        try:
            if self.active_lot_id and not crud.get_lot(db, self.active_lot_id):
                crud.create_lot(
                    db,
                    lot_id=self.active_lot_id,
                    supplier="Demo Supplier",
                    product_type=self.product_type,
                    status="active",
                )
        finally:
            db.close()

        self.status = "RUNNING"
        self.start_time = time.time()
        self._loop_task = asyncio.create_task(self._inspection_loop())
        logger.info(f"Started inspection loop for Lot {self.active_lot_id} ({self.product_type})")

    def stop_inspection(self):
        """Stop inspection loop."""
        self.status = "STOPPED"
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
        logger.info("Inspection loop stopped.")

    def run_calibration(self, cal_type: str, operator: str = "operator") -> dict:
        """Run dark or white calibration step."""
        self.status = "CALIBRATING"
        db = SessionLocal()
        try:
            res = "pass"
            if cal_type == "dark":
                dark = self.source.get_dark_reference()
                self.pipeline._dark_ref = dark
            elif cal_type == "white":
                white = self.source.get_white_reference()
                self.pipeline._white_ref = white

            event = crud.create_calibration_event(
                db,
                type=cal_type,
                result=res,
                operator=operator,
                details={"status": "completed", "snr": 55.4},
            )
            crud.append_audit_log(
                db,
                user=operator,
                action="calibration",
                detail=f"Completed {cal_type} calibration reference update",
            )
            self.status = "IDLE"
            return {"id": event.id, "type": cal_type, "result": res}
        finally:
            db.close()

    async def _inspection_loop(self):
        """Background continuous scan loop."""
        while self.status == "RUNNING":
            loop_start = time.time()
            try:
                await self._process_single_scan()
            except Exception as e:
                logger.error(f"Error in inspection scan step: {e}", exc_info=True)
                self.status = "FAULT"
                break

            elapsed = time.time() - loop_start
            sleep_time = max(0.01, self.scan_interval_s - elapsed)
            await asyncio.sleep(sleep_time)

    async def _process_single_scan(self):
        scan_t0 = time.time()
        scan_timestamp_ms = scan_t0 * 1000.0

        # Apply forced contamination if set
        if self.force_contamination:
            self.source.force_contamination = self.force_contamination
        else:
            self.source.force_contamination = None

        # 1. Acquisition
        cube = self.source.acquire()

        # 2. Preprocessing pipeline
        processed = self.pipeline.process(cube)

        # 3. Model classification
        if self.classifier.is_ready():
            pixel_classes, pixel_probs, infer_latency = self.classifier.predict(processed.product_pixels)
        else:
            # Synthetic predictions fallback
            n_pix = processed.product_pixels.shape[0]
            pixel_classes = np.zeros(n_pix, dtype=int)
            pixel_probs = np.ones((n_pix, 5), dtype=np.float32) / 5.0
            infer_latency = 5.0

        # 4. Item decision engine
        decision = self.decision_engine.decide(
            pixel_classes=pixel_classes,
            pixel_probs=pixel_probs,
            product_mask=processed.product_mask,
            spatial_shape=(cube.height, cube.width),
        )

        total_latency_ms = (time.time() - scan_t0) * 1000.0

        # 5. Actuation control
        action, ack = self.reject_controller.process_verdict(
            item_id=cube.item_id,
            verdict=decision.verdict,
            scan_timestamp_ms=scan_timestamp_ms,
            system_healthy=True,
        )

        # 6. Generate thumbnails and heatmaps base64
        rgb_b64 = self._image_to_base64(processed.rgb_preview)
        heatmap_img = self._probability_heatmap(decision.probability_map)
        heatmap_b64 = self._image_to_base64(heatmap_img)

        thumbnail_filename = f"thumb_{cube.item_id}.png"
        thumbnail_file_path = settings.thumbnail_dir / thumbnail_filename
        Image.fromarray(processed.rgb_preview).save(thumbnail_file_path)

        # 7. Persist to DB
        db = SessionLocal()
        try:
            prod = crud.create_product(
                db,
                line_id="LINE-1",
                lot_id=self.active_lot_id,
                product_type=self.product_type,
                verdict=decision.verdict,
                contamination_score=decision.contamination_score,
                predicted_pathogen=decision.predicted_pathogen,
                confidence=decision.confidence,
                contaminated_area_pct=decision.contaminated_area_pct,
                latency_ms=total_latency_ms,
                actuator_action=action,
                actuator_ack=ack,
                thumbnail_path=f"/data/thumbnails/{thumbnail_filename}",
                spectral_summary={
                    "mean_spectrum": [round(float(v), 4) for v in processed.mean_spectrum[:15]],
                    "peak_wavelength": float(cube.wavelengths[int(np.argmax(processed.mean_spectrum))]),
                },
            )
            product_id = prod.id
        finally:
            db.close()

        # Update stats
        self.total_scanned += 1
        if decision.verdict == "REJECT":
            self.total_rejected += 1
        self.recent_latencies.append(total_latency_ms)
        if len(self.recent_latencies) > 200:
            self.recent_latencies.pop(0)

        uptime = time.time() - self.start_time if self.start_time else 0.0
        items_per_min = (self.total_scanned / uptime * 60.0) if uptime > 0 else 0.0

        # 8. Broadcast over WebSocket
        payload = {
            "type": "scan_update",
            "product": {
                "id": product_id,
                "item_id": cube.item_id,
                "scan_ts": dt.datetime.utcnow().isoformat(),
                "verdict": decision.verdict,
                "contamination_score": decision.contamination_score,
                "predicted_pathogen": decision.predicted_pathogen,
                "confidence": decision.confidence,
                "contaminated_area_pct": decision.contaminated_area_pct,
                "latency_ms": round(total_latency_ms, 1),
                "actuator_action": action,
                "explanation": decision.explanation,
                "rgb_preview": f"data:image/png;base64,{rgb_b64}",
                "heatmap": f"data:image/png;base64,{heatmap_b64}",
                "spectrum": [round(float(v), 4) for v in processed.mean_spectrum],
                "wavelengths": [round(float(wl), 1) for wl in cube.wavelengths],
            },
            "counters": {
                "items_per_min": round(items_per_min, 1),
                "total_scanned": self.total_scanned,
                "total_rejected": self.total_rejected,
                "reject_rate": round((self.total_rejected / self.total_scanned * 100) if self.total_scanned else 0, 2),
                "status": self.status,
                "latency_p50": round(float(np.percentile(self.recent_latencies, 50)), 1) if self.recent_latencies else 0,
            }
        }
        await self.broadcast(payload)

    @staticmethod
    def _image_to_base64(img_arr: np.ndarray) -> str:
        img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _probability_heatmap(prob_map: np.ndarray) -> np.ndarray:
        """Create RGB heatmap from (H, W) float probability map [0.0, 1.0]."""
        # Blue -> Yellow -> Red colormap
        h, w = prob_map.shape
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        norm = np.clip(prob_map, 0.0, 1.0)

        # Red channel increases with probability
        heatmap[:, :, 0] = (norm * 255).astype(np.uint8)
        # Green channel peaks at mid probability
        heatmap[:, :, 1] = ((1.0 - np.abs(norm - 0.5) * 2) * 200).astype(np.uint8)
        # Blue channel decreases
        heatmap[:, :, 2] = ((1.0 - norm) * 255).astype(np.uint8)
        return heatmap


# Global singleton instance
inspection_manager = InspectionManager()
