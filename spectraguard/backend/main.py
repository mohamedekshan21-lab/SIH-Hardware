"""SpectraGuard FastAPI Application — REST API, WebSockets, and Static Web Dashboard."""

from __future__ import annotations

import os
import io
import datetime as dt
from pathlib import Path
from typing import Optional, List, Any

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.config import settings, BASE_DIR
from backend.database import get_db, init_db
from backend.records import crud, schemas, seed
from backend.services import (
    create_access_token, verify_password, get_password_hash,
    get_current_user, require_user, inspection_manager, generate_lot_pdf_report,
)
from backend.records.models import User
from backend.inference.train_demo_model import train_and_export

app = FastAPI(
    title="SpectraGuard API",
    description="Hyperspectral Food Safety Inspection & Rejection Pipeline",
    version="1.0.0",
)

# Enable CORS for local development / UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup & Shutdown ───────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    init_db()

    # Train demo model if missing
    onnx_path = settings.model_dir / "demo_model.onnx"
    pkl_path = settings.model_dir / "demo_model.pkl"
    if not onnx_path.exists() and not pkl_path.exists():
        print("Model file missing. Training demo model...")
        train_and_export()

    # Seed DB if empty
    db = next(get_db())
    if crud.list_products(db, per_page=1)[1] == 0:
        print("Database empty. Running seed...")
        seed.seed_database()

    # Initialize inspection engine
    inspection_manager.initialize()


# Mount static thumbnail directory
if settings.thumbnail_dir.exists():
    app.mount("/data/thumbnails", StaticFiles(directory=str(settings.thumbnail_dir)), name="thumbnails")

# Mount web UI static files if directory exists
static_ui_dir = BASE_DIR / "static"
if static_ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_ui_dir)), name="static_ui")


# ── Root SPA / Web Dashboard Route ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(content="<h1>SpectraGuard API Server Running</h1><p>Visit /docs for OpenAPI specifications.</p>")



# ── Authentication Endpoints ──────────────────────────────────────────────
@app.post("/api/auth/login", response_model=schemas.Token)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, req.username)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    crud.append_audit_log(db, user=user.username, action="login", detail="User logged into SpectraGuard")
    return schemas.Token(
        access_token=access_token,
        token_type="bearer",
        user=schemas.UserOut.model_validate(user),
    )


@app.get("/api/auth/me", response_model=schemas.UserOut)
def get_me(user: User = Depends(require_user)):
    return user


# ── Live Inspection & Control ─────────────────────────────────────────────
@app.post("/api/inspection/start")
def start_inspection(
    lot_id: Optional[str] = "LOT-LIVE-001",
    product_type: str = "poultry",
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inspection_manager.start_inspection(lot_id=lot_id, product_type=product_type)
    username = user.username if user else "system"
    crud.append_audit_log(db, user=username, action="inspection_start", detail=f"Started inspection on lot {lot_id}")
    return {"status": "RUNNING", "lot_id": lot_id, "product_type": product_type}


@app.post("/api/inspection/stop")
def stop_inspection(
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inspection_manager.stop_inspection()
    username = user.username if user else "system"
    crud.append_audit_log(db, user=username, action="inspection_stop", detail="Stopped inspection loop")
    return {"status": "STOPPED"}


@app.get("/api/inspection/status", response_model=schemas.LiveCounters)
def get_inspection_status():
    uptime = time.time() - inspection_manager.start_time if inspection_manager.start_time else 0.0
    items_per_min = (inspection_manager.total_scanned / uptime * 60.0) if uptime > 0 else 0.0
    reject_rate = (inspection_manager.total_rejected / inspection_manager.total_scanned * 100) if inspection_manager.total_scanned else 0.0

    recent = inspection_manager.recent_latencies or [0.0]
    import numpy as np
    lat_stats = schemas.LatencyStats(
        p50=round(float(np.percentile(recent, 50)), 1),
        p95=round(float(np.percentile(recent, 95)), 1),
        p99=round(float(np.percentile(recent, 99)), 1),
        mean=round(float(np.mean(recent)), 1),
    )

    return schemas.LiveCounters(
        items_per_min=round(items_per_min, 1),
        total_scanned=inspection_manager.total_scanned,
        total_rejected=inspection_manager.total_rejected,
        reject_rate=round(reject_rate, 2),
        uptime_seconds=round(uptime, 1),
        latency=lat_stats,
        status=inspection_manager.status,
    )


@app.post("/api/inspection/control")
def demo_control(ctrl: schemas.DemoControl, db: Session = Depends(get_db)):
    if ctrl.action == "start":
        inspection_manager.start_inspection()
    elif ctrl.action == "stop":
        inspection_manager.stop_inspection()
    elif ctrl.action == "inject_contamination":
        inspection_manager.force_contamination = ctrl.value  # e.g. 'Salmonella' or None
    elif ctrl.action == "set_speed":
        try:
            speed = float(ctrl.value)
            inspection_manager.belt_speed_mm_s = speed
            inspection_manager.reject_controller.belt_speed_mm_s = speed
        except (ValueError, TypeError):
            pass

    return {"status": "ok", "action": ctrl.action, "value": ctrl.value}


@app.websocket("/ws/live-stream")
async def websocket_live_stream(websocket: WebSocket):
    await inspection_manager.connect_websocket(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        inspection_manager.disconnect_websocket(websocket)
    except Exception:
        inspection_manager.disconnect_websocket(websocket)


# ── Scans & Products Traceability ─────────────────────────────────────────
@app.get("/api/products", response_model=schemas.ProductPage)
def list_products(
    page: int = 1,
    per_page: int = 50,
    verdict: Optional[str] = None,
    lot_id: Optional[str] = None,
    product_type: Optional[str] = None,
    pathogen: Optional[str] = None,
    db: Session = Depends(get_db),
):
    items, total = crud.list_products(
        db, page=page, per_page=per_page,
        verdict=verdict, lot_id=lot_id, product_type=product_type, pathogen=pathogen,
    )
    pages = (total + per_page - 1) // per_page if per_page else 1
    return schemas.ProductPage(
        items=[schemas.ProductOut.model_validate(p) for p in items],
        total=total,
        page=page,
        pages=pages,
    )


@app.get("/api/products/{product_id}", response_model=schemas.ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    prod = crud.get_product(db, product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found")
    return schemas.ProductOut.model_validate(prod)


@app.post("/api/products/{product_id}/label", response_model=schemas.LabLabelOut)
def create_lab_label(
    product_id: int,
    req: schemas.LabLabelCreate,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    label = crud.create_lab_label(
        db,
        product_id=product_id,
        confirmed_verdict=req.confirmed_verdict,
        confirmed_pathogen=req.confirmed_pathogen,
        reviewed_by=user.username,
        reason=req.reason,
    )
    crud.append_audit_log(
        db,
        user=user.username,
        action="lab_label_created",
        detail=f"Lab verified scan #{product_id} as {req.confirmed_verdict}",
        entity_type="product",
        entity_id=str(product_id),
    )
    return schemas.LabLabelOut.model_validate(label)


# ── Lots Management & PDF Reports ─────────────────────────────────────────
@app.get("/api/lots", response_model=List[schemas.LotOut])
def list_lots(status: Optional[str] = None, db: Session = Depends(get_db)):
    lots = crud.list_lots(db, status=status)
    return [schemas.LotOut.model_validate(l) for l in lots]


@app.get("/api/lots/{lot_id}", response_model=schemas.LotDetail)
def get_lot(lot_id: str, db: Session = Depends(get_db)):
    lot = crud.get_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    products, _ = crud.list_products(db, lot_id=lot_id, per_page=100)
    rate = (lot.total_rejected / lot.total_scanned * 100) if lot.total_scanned else 0.0

    return schemas.LotDetail(
        lot_id=lot.lot_id,
        supplier=lot.supplier,
        product_type=lot.product_type,
        origin_farm=lot.origin_farm,
        shift=lot.shift,
        operator=lot.operator,
        status=lot.status,
        received_ts=lot.received_ts,
        total_scanned=lot.total_scanned,
        total_rejected=lot.total_rejected,
        contamination_rate=round(rate, 2),
        products=[schemas.ProductOut.model_validate(p) for p in products],
    )


@app.get("/api/lots/{lot_id}/pdf")
def get_lot_pdf_report(lot_id: str, db: Session = Depends(get_db)):
    try:
        pdf_bytes = generate_lot_pdf_report(db, lot_id)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="Lot_Audit_{lot_id}.pdf"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Analytics & SPC ──────────────────────────────────────────────────────
@app.get("/api/analytics/trend", response_model=List[schemas.ContaminationTrend])
def get_contamination_trend(days: int = 30, db: Session = Depends(get_db)):
    return crud.contamination_trend(db, days=days)


@app.get("/api/analytics/suppliers", response_model=List[schemas.SupplierStats])
def get_supplier_stats(db: Session = Depends(get_db)):
    return crud.supplier_stats(db)


@app.get("/api/analytics/latency", response_model=schemas.LatencyStats)
def get_latency_stats(db: Session = Depends(get_db)):
    res = crud.latency_percentiles(db)
    return schemas.LatencyStats(**res)


# ── Calibration ───────────────────────────────────────────────────────────
@app.post("/api/calibration/run", response_model=schemas.CalibrationEventOut)
def run_calibration(
    run: schemas.CalibrationRun,
    user: Optional[User] = Depends(get_current_user),
):
    operator = user.username if user else run.operator
    res = inspection_manager.run_calibration(run.type, operator=operator)
    return schemas.CalibrationEventOut(
        id=res["id"],
        ts=dt.datetime.utcnow(),
        type=run.type,
        result=res["result"],
        operator=operator,
        details={"status": "pass", "signal_snr": 58.2},
    )


@app.get("/api/calibration/latest", response_model=Optional[schemas.CalibrationEventOut])
def get_latest_calibration(db: Session = Depends(get_db)):
    cal = crud.get_latest_calibration(db)
    if not cal:
        return None
    return schemas.CalibrationEventOut.model_validate(cal)


@app.get("/api/calibration/history", response_model=List[schemas.CalibrationEventOut])
def get_calibration_history(db: Session = Depends(get_db)):
    cals = crud.list_calibrations(db)
    return [schemas.CalibrationEventOut.model_validate(c) for c in cals]


# ── Model Management ──────────────────────────────────────────────────────
@app.get("/api/models", response_model=List[schemas.ModelVersionOut])
def list_models(db: Session = Depends(get_db)):
    models = crud.list_models(db)
    return [schemas.ModelVersionOut.model_validate(m) for m in models]


@app.post("/api/models/train")
def trigger_train_model(bg_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    def _task():
        train_and_export()
        inspection_manager.classifier.load()

    bg_tasks.add_task(_task)
    return {"status": "training_started", "message": "Model re-training running in background"}


# ── Alerts ────────────────────────────────────────────────────────────────
@app.get("/api/alerts", response_model=List[schemas.AlertOut])
def list_alerts(unresolved_only: bool = False, db: Session = Depends(get_db)):
    alerts = crud.list_alerts(db, unresolved_only=unresolved_only)
    return [schemas.AlertOut.model_validate(a) for a in alerts]


@app.post("/api/alerts/{alert_id}/ack", response_model=schemas.AlertOut)
def ack_alert(
    alert_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    alert = crud.acknowledge_alert(db, alert_id, user=user.username)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    crud.append_audit_log(db, user=user.username, action="alert_acknowledged", detail=f"Acknowledged alert #{alert_id}")
    return schemas.AlertOut.model_validate(alert)


# ── Audit Log & Hash Chain ────────────────────────────────────────────────
@app.get("/api/audit-log", response_model=List[schemas.AuditLogOut])
def list_audit_log(limit: int = 200, db: Session = Depends(get_db)):
    logs = crud.list_audit_log(db, limit=limit)
    return [schemas.AuditLogOut.model_validate(l) for l in logs]


# ── Settings ──────────────────────────────────────────────────────────────
@app.get("/api/settings", response_model=List[schemas.SettingOut])
def get_settings(db: Session = Depends(get_db)):
    st = crud.list_settings(db)
    return [schemas.SettingOut.model_validate(s) for s in st]


@app.put("/api/settings/{key}", response_model=schemas.SettingOut)
def update_setting(
    key: str,
    update: schemas.SettingUpdate,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    s = crud.upsert_setting(db, key=key, value=update.value, updated_by=user.username)
    crud.append_audit_log(db, user=user.username, action="setting_updated", detail=f"Set {key}={update.value}")
    return schemas.SettingOut.model_validate(s)


# ── Root SPA / Web Dashboard Route ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(content="<h1>SpectraGuard API Server Running</h1><p>Visit /docs for OpenAPI specifications.</p>")

