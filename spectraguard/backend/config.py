"""SpectraGuard — configuration loaded from .env."""

from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Database
    database_url: str = f"sqlite:///{BASE_DIR / 'spectraguard.db'}"

    # Auth / JWT
    secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 480
    default_admin_password: str = "spectraguard-admin"

    # MQTT
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_reject: str = "spectraguard/actuator/reject"
    mqtt_topic_ack: str = "spectraguard/actuator/ack"

    # Conveyor / timing
    belt_speed_mm_s: float = 500.0
    reject_delay_ms: int = 150
    scan_interval_ms: int = 200

    # Calibration
    calibration_max_age_hours: int = 8

    # Logging
    log_level: str = "INFO"

    # Demo / simulator
    demo_mode: bool = True
    simulated_items_per_min: int = 30
    contamination_rate: float = 0.08

    # Paths
    model_dir: Path = BASE_DIR / "inference" / "models"
    thumbnail_dir: Path = BASE_DIR / "data" / "thumbnails"

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure dirs exist
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.thumbnail_dir.mkdir(parents=True, exist_ok=True)
