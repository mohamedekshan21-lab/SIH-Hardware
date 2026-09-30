"""Pydantic schemas for API serialization."""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
class ProductBase(BaseModel):
    line_id: str = "LINE-1"
    lot_id: Optional[str] = None
    product_type: str = "poultry"
    verdict: str = "REVIEW"
    contamination_score: float = 0.0
    predicted_pathogen: Optional[str] = None
    confidence: float = 0.0
    contaminated_area_pct: float = 0.0
    latency_ms: float = 0.0
    actuator_action: Optional[str] = None
    actuator_ack: Optional[bool] = None
    thumbnail_path: Optional[str] = None
    spectral_summary: Optional[dict] = None


class ProductCreate(ProductBase):
    pass


class ProductOut(ProductBase):
    id: int
    scan_ts: dt.datetime

    class Config:
        from_attributes = True


class ProductPage(BaseModel):
    items: List[ProductOut]
    total: int
    page: int
    pages: int


# ---------------------------------------------------------------------------
# Lots
# ---------------------------------------------------------------------------
class LotBase(BaseModel):
    lot_id: str
    supplier: Optional[str] = None
    product_type: str = "poultry"
    origin_farm: Optional[str] = None
    shift: Optional[str] = None
    operator: Optional[str] = None
    status: str = "active"


class LotCreate(LotBase):
    pass


class LotOut(LotBase):
    received_ts: dt.datetime
    total_scanned: int = 0
    total_rejected: int = 0

    class Config:
        from_attributes = True


class LotDetail(LotOut):
    products: List[ProductOut] = []
    contamination_rate: float = 0.0


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
class AlertOut(BaseModel):
    id: int
    ts: dt.datetime
    severity: str
    type: str
    message: str
    acknowledged_by: Optional[str] = None
    acknowledged_ts: Optional[dt.datetime] = None
    resolved: bool = False

    class Config:
        from_attributes = True


class AlertAcknowledge(BaseModel):
    acknowledged_by: str


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class CalibrationEventOut(BaseModel):
    id: int
    ts: dt.datetime
    type: str
    result: str
    operator: Optional[str] = None
    details: Optional[dict] = None

    class Config:
        from_attributes = True


class CalibrationRun(BaseModel):
    type: str  # dark | white | wavelength
    operator: str = "system"


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------
class ModelVersionOut(BaseModel):
    id: int
    name: str
    trained_ts: dt.datetime
    metrics: Optional[dict] = None
    active: bool = False
    path: Optional[str] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Users & Auth
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    name: str
    role: str = "Operator"


class UserOut(BaseModel):
    id: int
    username: str
    name: str
    role: str
    active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
class AuditLogOut(BaseModel):
    id: int
    ts: dt.datetime
    user: str
    action: str
    detail: Optional[str] = None
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    entry_hash: str

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class SettingOut(BaseModel):
    key: str
    value: str
    updated_ts: dt.datetime
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True


class SettingUpdate(BaseModel):
    value: str
    updated_by: str = "system"


# ---------------------------------------------------------------------------
# Lab labels / QA review
# ---------------------------------------------------------------------------
class LabLabelCreate(BaseModel):
    product_id: int
    confirmed_verdict: str
    confirmed_pathogen: Optional[str] = None
    reviewed_by: str
    reason: Optional[str] = None


class LabLabelOut(LabLabelCreate):
    id: int
    reviewed_ts: dt.datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
class ContaminationTrend(BaseModel):
    date: str
    total: int
    rejected: int
    rate: float


class SupplierStats(BaseModel):
    supplier: str
    total: int
    rejected: int
    rate: float


class LatencyStats(BaseModel):
    p50: float
    p95: float
    p99: float
    mean: float


class LiveCounters(BaseModel):
    items_per_min: float = 0.0
    total_scanned: int = 0
    total_rejected: int = 0
    reject_rate: float = 0.0
    uptime_seconds: float = 0.0
    latency: LatencyStats = LatencyStats(p50=0, p95=0, p99=0, mean=0)
    status: str = "IDLE"  # RUNNING | CALIBRATION_NEEDED | FAULT | IDLE


class DemoControl(BaseModel):
    action: str  # start | stop | inject_contamination | set_speed
    value: Optional[Any] = None
