"""SpectraGuard ORM models — full data model for traceability."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer,
    String, Text, JSON, func,
)
from sqlalchemy.orm import relationship

from backend.database import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Verdict(str):
    PASS = "PASS"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class Severity(str):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class UserRole(str):
    OPERATOR = "Operator"
    QA_MANAGER = "QA Manager"
    ADMIN = "Admin"


# ---------------------------------------------------------------------------
# Products — every scanned item
# ---------------------------------------------------------------------------
class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_ts = Column(DateTime, default=dt.datetime.utcnow, index=True)
    line_id = Column(String(32), default="LINE-1")
    lot_id = Column(String(64), ForeignKey("lots.lot_id"), nullable=True, index=True)
    product_type = Column(String(32), default="poultry")  # poultry | leafy_greens
    verdict = Column(String(10), default=Verdict.REVIEW, index=True)
    contamination_score = Column(Float, default=0.0)
    predicted_pathogen = Column(String(32), nullable=True)
    confidence = Column(Float, default=0.0)
    contaminated_area_pct = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    actuator_action = Column(String(16), nullable=True)
    actuator_ack = Column(Boolean, nullable=True)
    thumbnail_path = Column(String(256), nullable=True)
    spectral_summary = Column(JSON, nullable=True)

    lot = relationship("Lot", back_populates="products")


# ---------------------------------------------------------------------------
# Lots — grouping of products by lot
# ---------------------------------------------------------------------------
class Lot(Base):
    __tablename__ = "lots"

    lot_id = Column(String(64), primary_key=True)
    supplier = Column(String(128), nullable=True)
    product_type = Column(String(32), default="poultry")
    received_ts = Column(DateTime, default=dt.datetime.utcnow)
    origin_farm = Column(String(128), nullable=True)
    shift = Column(String(16), nullable=True)
    operator = Column(String(64), nullable=True)
    total_scanned = Column(Integer, default=0)
    total_rejected = Column(Integer, default=0)
    status = Column(String(20), default="active")  # active | released | hold | recalled

    products = relationship("Product", back_populates="lot")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=dt.datetime.utcnow, index=True)
    severity = Column(String(10), default=Severity.INFO)
    type = Column(String(64), default="general")
    message = Column(Text, default="")
    acknowledged_by = Column(String(64), nullable=True)
    acknowledged_ts = Column(DateTime, nullable=True)
    resolved = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# Calibration events
# ---------------------------------------------------------------------------
class CalibrationEvent(Base):
    __tablename__ = "calibration_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=dt.datetime.utcnow)
    type = Column(String(32))  # dark | white | wavelength
    result = Column(String(16))  # pass | fail
    operator = Column(String(64), nullable=True)
    details = Column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128))
    trained_ts = Column(DateTime, default=dt.datetime.utcnow)
    metrics = Column(JSON, nullable=True)
    active = Column(Boolean, default=False)
    path = Column(String(256), nullable=True)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True)
    hashed_password = Column(String(256))
    name = Column(String(128))
    role = Column(String(20), default=UserRole.OPERATOR)
    active = Column(Boolean, default=True)


# ---------------------------------------------------------------------------
# Audit log — immutable, hash-chained
# ---------------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=dt.datetime.utcnow)
    user = Column(String(64))
    action = Column(String(128))
    detail = Column(Text, nullable=True)
    entity_type = Column(String(32), nullable=True)
    entity_id = Column(String(64), nullable=True)
    prev_hash = Column(String(64), default="0" * 64)
    entry_hash = Column(String(64))

    @staticmethod
    def compute_hash(prev_hash: str, ts: str, user: str, action: str, detail: str) -> str:
        payload = f"{prev_hash}|{ts}|{user}|{action}|{detail}"
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Settings (key-value store for runtime config)
# ---------------------------------------------------------------------------
class SettingEntry(Base):
    __tablename__ = "settings"

    key = Column(String(128), primary_key=True)
    value = Column(Text)
    updated_ts = Column(DateTime, default=dt.datetime.utcnow)
    updated_by = Column(String(64), nullable=True)


# ---------------------------------------------------------------------------
# Lab-confirmed labels (for model improvement)
# ---------------------------------------------------------------------------
class LabLabel(Base):
    __tablename__ = "lab_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    confirmed_verdict = Column(String(10))
    confirmed_pathogen = Column(String(32), nullable=True)
    reviewed_by = Column(String(64))
    reviewed_ts = Column(DateTime, default=dt.datetime.utcnow)
    reason = Column(Text, nullable=True)
