"""Seed the database with ~5000 demo scans across 30 days.

Tells a story: supplier "GreenValley Farms" has a Listeria cluster
starting around day 20, eventually triggering alerts and a hold.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
import json
import numpy as np
from backend.services.auth_service import get_password_hash
from sqlalchemy.orm import Session
from backend.database import SessionLocal, init_db
from backend.records.models import (
    Product, Lot, Alert, CalibrationEvent, ModelVersion,
    User, AuditLog, SettingEntry,
)

SUPPLIERS = [
    ("GreenValley Farms", "Lot-GV-{i:03d}", "poultry", "Midwest Region"),
    ("FreshLeaf Co.", "Lot-FL-{i:03d}", "leafy_greens", "California"),
    ("PrimeCut Inc.", "Lot-PC-{i:03d}", "poultry", "Southeast"),
    ("OrganicGreens LLC", "Lot-OG-{i:03d}", "leafy_greens", "Oregon"),
]

PATHOGENS = ["Salmonella", "Listeria", "E. coli", "Biofilm"]
SHIFTS = ["morning", "afternoon", "night"]
LINES = ["LINE-1", "LINE-2"]
OPERATORS = ["jsmith", "mgarcia", "awong", "bpatel"]

random.seed(42)
np.random.seed(42)


def _spectral_summary(contaminated: bool, pathogen: str | None = None) -> dict:
    """Generate a plausible spectral summary blob."""
    bands = np.linspace(900, 1700, 224).tolist()
    base = np.exp(-0.5 * ((np.array(bands) - 1200) / 200) ** 2) * 0.8
    if contaminated and pathogen:
        # Add pathogen-specific absorption dip
        center = {"Salmonella": 1050, "Listeria": 1150, "E. coli": 980, "Biofilm": 1300}
        c = center.get(pathogen, 1100)
        dip = -0.15 * np.exp(-0.5 * ((np.array(bands) - c) / 30) ** 2)
        base += dip
    base += np.random.normal(0, 0.01, len(bands))
    return {
        "mean_spectrum": [round(float(v), 4) for v in base[:10]],  # truncated for DB
        "peak_wavelength": round(float(bands[int(np.argmax(base))]), 1),
        "min_reflectance": round(float(np.min(base)), 4),
    }


def seed_database():
    """Run the full seed process."""
    init_db()
    db = SessionLocal()

    # Check if already seeded
    if db.query(Product).count() > 100:
        print("Database already seeded — skipping.")
        db.close()
        return

    print("Seeding database...")

    # ── Users ──
    users_data = [
        ("admin", "admin123", "System Admin", "Admin"),
        ("jsmith", "pass123", "John Smith", "Operator"),
        ("mgarcia", "pass123", "Maria Garcia", "Operator"),
        ("awong", "pass123", "Alice Wong", "QA Manager"),
        ("bpatel", "pass123", "Bob Patel", "QA Manager"),
    ]
    for uname, pwd, name, role in users_data:
        if not db.query(User).filter(User.username == uname).first():
            db.add(User(
                username=uname,
                hashed_password=get_password_hash(pwd),
                name=name,
                role=role,
            ))

    db.commit()

    # ── Default settings ──
    defaults = {
        "contamination_threshold": "0.5",
        "reject_confidence_min": "0.7",
        "belt_speed_mm_s": "500",
        "reject_delay_ms": "150",
        "calibration_max_age_hours": "8",
        "spc_control_limit": "0.12",
        "active_product_profile": "poultry",
    }
    for k, v in defaults.items():
        if not db.query(SettingEntry).filter(SettingEntry.key == k).first():
            db.add(SettingEntry(key=k, value=v, updated_by="seed"))
    db.commit()

    # ── Model version ──
    if not db.query(ModelVersion).first():
        db.add(ModelVersion(
            name="demo-1dcnn-v1",
            metrics={
                "accuracy": 0.94, "sensitivity": 0.91, "specificity": 0.96,
                "f1": 0.92, "fnr": 0.09, "auc": 0.97,
            },
            active=True,
            path="models/demo_model.onnx",
        ))
    db.commit()

    # ── Lots ──
    now = dt.datetime.utcnow()
    lots_created = []
    lot_counter = 0
    for day_offset in range(30):
        day = now - dt.timedelta(days=29 - day_offset)
        # 2-4 lots per day
        for _ in range(random.randint(2, 4)):
            lot_counter += 1
            supplier_info = random.choice(SUPPLIERS)
            supplier_name, lot_tmpl, prod_type, origin = supplier_info
            lot_id = lot_tmpl.format(i=lot_counter)
            shift = random.choice(SHIFTS)
            operator = random.choice(OPERATORS)
            lot = Lot(
                lot_id=lot_id,
                supplier=supplier_name,
                product_type=prod_type,
                received_ts=day + dt.timedelta(hours=random.randint(6, 18)),
                origin_farm=origin,
                shift=shift,
                operator=operator,
                status="released",
            )
            db.add(lot)
            lots_created.append((lot_id, supplier_name, prod_type, day_offset, day))
    db.commit()

    # ── Products (5000 items) ──
    products_per_lot = 5000 // len(lots_created) + 1
    product_count = 0

    # GreenValley contamination cluster: days 20-26
    gv_contamination_days = set(range(20, 27))

    for lot_id, supplier, prod_type, day_offset, day in lots_created:
        n_items = random.randint(products_per_lot - 10, products_per_lot + 10)
        n_items = min(n_items, 5000 - product_count)
        if n_items <= 0:
            break

        is_gv_cluster = supplier == "GreenValley Farms" and day_offset in gv_contamination_days
        base_contam_rate = 0.25 if is_gv_cluster else 0.06

        lot_scanned = 0
        lot_rejected = 0

        for j in range(n_items):
            product_count += 1
            scan_time = day + dt.timedelta(
                hours=random.randint(6, 22),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            contaminated = random.random() < base_contam_rate
            if contaminated:
                pathogen = "Listeria" if is_gv_cluster else random.choice(PATHOGENS)
                score = round(random.uniform(0.55, 0.98), 3)
                confidence = round(random.uniform(0.6, 0.99), 3)
                area_pct = round(random.uniform(2, 35), 1)
                verdict = "REJECT" if confidence > 0.7 else "REVIEW"
            else:
                pathogen = None
                score = round(random.uniform(0.01, 0.35), 3)
                confidence = round(random.uniform(0.85, 0.99), 3)
                area_pct = 0.0
                verdict = "PASS"

            latency = round(random.gauss(45, 12), 1)
            latency = max(8, min(latency, 150))

            lot_scanned += 1
            if verdict == "REJECT":
                lot_rejected += 1

            db.add(Product(
                scan_ts=scan_time,
                line_id=random.choice(LINES),
                lot_id=lot_id,
                product_type=prod_type,
                verdict=verdict,
                contamination_score=score,
                predicted_pathogen=pathogen,
                confidence=confidence,
                contaminated_area_pct=area_pct,
                latency_ms=latency,
                actuator_action="reject" if verdict == "REJECT" else ("hold" if verdict == "REVIEW" else "pass"),
                actuator_ack=True if verdict in ("REJECT", "REVIEW") else None,
                spectral_summary=_spectral_summary(contaminated, pathogen),
            ))

        # Update lot counts
        lot_obj = db.query(Lot).filter(Lot.lot_id == lot_id).first()
        if lot_obj:
            lot_obj.total_scanned = lot_scanned
            lot_obj.total_rejected = lot_rejected
            if is_gv_cluster:
                lot_obj.status = "hold"

    db.commit()
    print(f"  Seeded {product_count} products across {len(lots_created)} lots.")

    # ── Alerts for the contamination cluster ──
    alerts_data = [
        (now - dt.timedelta(days=10), "WARNING", "contamination_rate",
         "Rolling contamination rate for LINE-1 exceeded 10% control limit."),
        (now - dt.timedelta(days=9), "CRITICAL", "cluster_detected",
         "Listeria cluster detected: GreenValley Farms lots show 25% reject rate over 3 consecutive lots."),
        (now - dt.timedelta(days=9, hours=2), "CRITICAL", "lot_hold",
         "Lots from GreenValley Farms placed on HOLD pending lab confirmation."),
        (now - dt.timedelta(days=8), "WARNING", "calibration",
         "Calibration reminder: white reference is 6 hours old."),
        (now - dt.timedelta(days=5), "INFO", "model_performance",
         "Model sensitivity check: 91.2% on last 500 scans, within spec."),
    ]
    for ts, sev, atype, msg in alerts_data:
        db.add(Alert(ts=ts, severity=sev, type=atype, message=msg))
    db.commit()

    # ── Calibration events ──
    for d in range(0, 30, 2):
        cal_ts = now - dt.timedelta(days=29 - d, hours=6)
        for cal_type in ["dark", "white"]:
            db.add(CalibrationEvent(
                ts=cal_ts, type=cal_type, result="pass",
                operator=random.choice(OPERATORS),
                details={"snr": round(random.uniform(45, 60), 1)},
            ))
    db.commit()

    # ── Initial audit log entries ──
    prev_hash = "0" * 64
    audit_entries = [
        ("admin", "system_start", "SpectraGuard system initialized"),
        ("admin", "model_activated", "Activated demo-1dcnn-v1"),
        ("awong", "lot_hold", "Held GreenValley Farms lots for lab review"),
    ]
    for user, action, detail in audit_entries:
        ts = dt.datetime.utcnow().isoformat()
        entry_hash = AuditLog.compute_hash(prev_hash, ts, user, action, detail)
        db.add(AuditLog(
            user=user, action=action, detail=detail,
            prev_hash=prev_hash, entry_hash=entry_hash,
        ))
        prev_hash = entry_hash
    db.commit()

    print("  Seed complete.")
    db.close()


if __name__ == "__main__":
    seed_database()
