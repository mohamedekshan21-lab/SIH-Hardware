"""CRUD operations for all SpectraGuard entities."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

from sqlalchemy import func, desc, Integer
from sqlalchemy.orm import Session

from backend.records.models import (
    Product, Lot, Alert, CalibrationEvent, ModelVersion,
    User, AuditLog, SettingEntry, LabLabel,
)


# ── Products ──────────────────────────────────────────────────────────────
def create_product(db: Session, **kwargs) -> Product:
    p = Product(**kwargs)
    db.add(p)
    db.commit()
    db.refresh(p)
    # Update lot counts
    if p.lot_id:
        lot = db.query(Lot).filter(Lot.lot_id == p.lot_id).first()
        if lot:
            lot.total_scanned += 1
            if p.verdict == "REJECT":
                lot.total_rejected += 1
            db.commit()
    return p


def get_product(db: Session, product_id: int) -> Optional[Product]:
    return db.query(Product).filter(Product.id == product_id).first()


def list_products(
    db: Session,
    page: int = 1,
    per_page: int = 50,
    verdict: Optional[str] = None,
    lot_id: Optional[str] = None,
    product_type: Optional[str] = None,
    pathogen: Optional[str] = None,
    line_id: Optional[str] = None,
    date_from: Optional[dt.datetime] = None,
    date_to: Optional[dt.datetime] = None,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
) -> Tuple[List[Product], int]:
    q = db.query(Product)
    if verdict:
        q = q.filter(Product.verdict == verdict)
    if lot_id:
        q = q.filter(Product.lot_id == lot_id)
    if product_type:
        q = q.filter(Product.product_type == product_type)
    if pathogen:
        q = q.filter(Product.predicted_pathogen == pathogen)
    if line_id:
        q = q.filter(Product.line_id == line_id)
    if date_from:
        q = q.filter(Product.scan_ts >= date_from)
    if date_to:
        q = q.filter(Product.scan_ts <= date_to)
    if score_min is not None:
        q = q.filter(Product.contamination_score >= score_min)
    if score_max is not None:
        q = q.filter(Product.contamination_score <= score_max)
    total = q.count()
    items = q.order_by(desc(Product.scan_ts)).offset((page - 1) * per_page).limit(per_page).all()
    return items, total


# ── Lots ──────────────────────────────────────────────────────────────────
def create_lot(db: Session, **kwargs) -> Lot:
    lot = Lot(**kwargs)
    db.add(lot)
    db.commit()
    db.refresh(lot)
    return lot


def get_lot(db: Session, lot_id: str) -> Optional[Lot]:
    return db.query(Lot).filter(Lot.lot_id == lot_id).first()


def list_lots(db: Session, status: Optional[str] = None) -> List[Lot]:
    q = db.query(Lot)
    if status:
        q = q.filter(Lot.status == status)
    return q.order_by(desc(Lot.received_ts)).all()


# ── Alerts ────────────────────────────────────────────────────────────────
def create_alert(db: Session, severity: str, alert_type: str, message: str) -> Alert:
    a = Alert(severity=severity, type=alert_type, message=message)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def list_alerts(db: Session, unresolved_only: bool = False) -> List[Alert]:
    q = db.query(Alert)
    if unresolved_only:
        q = q.filter(Alert.resolved == False)
    return q.order_by(desc(Alert.ts)).limit(200).all()


def acknowledge_alert(db: Session, alert_id: int, user: str) -> Optional[Alert]:
    a = db.query(Alert).filter(Alert.id == alert_id).first()
    if a:
        a.acknowledged_by = user
        a.acknowledged_ts = dt.datetime.utcnow()
        a.resolved = True
        db.commit()
        db.refresh(a)
    return a


# ── Calibration ───────────────────────────────────────────────────────────
def create_calibration_event(db: Session, **kwargs) -> CalibrationEvent:
    c = CalibrationEvent(**kwargs)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def get_latest_calibration(db: Session) -> Optional[CalibrationEvent]:
    return db.query(CalibrationEvent).order_by(desc(CalibrationEvent.ts)).first()


def list_calibrations(db: Session) -> List[CalibrationEvent]:
    return db.query(CalibrationEvent).order_by(desc(CalibrationEvent.ts)).limit(50).all()


# ── Model versions ────────────────────────────────────────────────────────
def get_active_model(db: Session) -> Optional[ModelVersion]:
    return db.query(ModelVersion).filter(ModelVersion.active == True).first()


def list_models(db: Session) -> List[ModelVersion]:
    return db.query(ModelVersion).order_by(desc(ModelVersion.trained_ts)).all()


def set_active_model(db: Session, model_id: int) -> Optional[ModelVersion]:
    db.query(ModelVersion).update({ModelVersion.active: False})
    m = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if m:
        m.active = True
        db.commit()
        db.refresh(m)
    return m


# ── Users ─────────────────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, **kwargs) -> User:
    u = User(**kwargs)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def list_users(db: Session) -> List[User]:
    return db.query(User).all()


# ── Audit log ─────────────────────────────────────────────────────────────
def append_audit_log(
    db: Session,
    user: str,
    action: str,
    detail: str = "",
    entity_type: str | None = None,
    entity_id: str | None = None,
) -> AuditLog:
    # Get previous hash
    last = db.query(AuditLog).order_by(desc(AuditLog.id)).first()
    prev_hash = last.entry_hash if last else "0" * 64
    ts = dt.datetime.utcnow().isoformat()
    entry_hash = AuditLog.compute_hash(prev_hash, ts, user, action, detail)
    entry = AuditLog(
        user=user, action=action, detail=detail,
        entity_type=entity_type, entity_id=entity_id,
        prev_hash=prev_hash, entry_hash=entry_hash,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_audit_log(db: Session, limit: int = 200) -> List[AuditLog]:
    return db.query(AuditLog).order_by(desc(AuditLog.ts)).limit(limit).all()


# ── Settings ──────────────────────────────────────────────────────────────
def get_setting(db: Session, key: str) -> Optional[SettingEntry]:
    return db.query(SettingEntry).filter(SettingEntry.key == key).first()


def upsert_setting(db: Session, key: str, value: str, updated_by: str = "system") -> SettingEntry:
    s = db.query(SettingEntry).filter(SettingEntry.key == key).first()
    if s:
        s.value = value
        s.updated_by = updated_by
        s.updated_ts = dt.datetime.utcnow()
    else:
        s = SettingEntry(key=key, value=value, updated_by=updated_by)
        db.add(s)
    db.commit()
    db.refresh(s)
    return s


def list_settings(db: Session) -> List[SettingEntry]:
    return db.query(SettingEntry).all()


# ── Lab labels ────────────────────────────────────────────────────────────
def create_lab_label(db: Session, **kwargs) -> LabLabel:
    ll = LabLabel(**kwargs)
    db.add(ll)
    db.commit()
    db.refresh(ll)
    return ll


def list_lab_labels(db: Session, product_id: Optional[int] = None) -> List[LabLabel]:
    q = db.query(LabLabel)
    if product_id:
        q = q.filter(LabLabel.product_id == product_id)
    return q.order_by(desc(LabLabel.reviewed_ts)).all()


# ── Analytics queries ─────────────────────────────────────────────────────
def contamination_trend(db: Session, days: int = 30):
    """Daily contamination rate over the last N days."""
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    rows = (
        db.query(
            func.date(Product.scan_ts).label("date"),
            func.count().label("total"),
            func.sum(func.cast(Product.verdict == "REJECT", Integer)).label("rejected"),
        )
        .filter(Product.scan_ts >= cutoff)
        .group_by(func.date(Product.scan_ts))
        .order_by(func.date(Product.scan_ts))
        .all()
    )
    return [
        {"date": str(r.date), "total": r.total, "rejected": r.rejected or 0,
         "rate": round((r.rejected or 0) / r.total, 4) if r.total else 0}
        for r in rows
    ]


def supplier_stats(db: Session):
    """Contamination rate by supplier."""
    rows = (
        db.query(
            Lot.supplier,
            func.count(Product.id).label("total"),
            func.sum(func.cast(Product.verdict == "REJECT", Integer)).label("rejected"),
        )
        .join(Product, Product.lot_id == Lot.lot_id)
        .group_by(Lot.supplier)
        .all()
    )
    return [
        {"supplier": r.supplier or "Unknown", "total": r.total,
         "rejected": r.rejected or 0,
         "rate": round((r.rejected or 0) / r.total, 4) if r.total else 0}
        for r in rows
    ]


def latency_percentiles(db: Session):
    """Compute p50/p95/p99 latency from recent scans."""
    rows = (
        db.query(Product.latency_ms)
        .order_by(desc(Product.scan_ts))
        .limit(1000)
        .all()
    )
    if not rows:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
    import numpy as np
    vals = np.array([r.latency_ms for r in rows])
    return {
        "p50": round(float(np.percentile(vals, 50)), 2),
        "p95": round(float(np.percentile(vals, 95)), 2),
        "p99": round(float(np.percentile(vals, 99)), 2),
        "mean": round(float(np.mean(vals)), 2),
    }
