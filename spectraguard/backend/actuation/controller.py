"""RejectController orchestrates diverter triggering, belt speed/timing compensation, dry-run mode, and fail-safe logic."""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from backend.actuation.simulated_driver import SimulatedDriver

logger = logging.getLogger(__name__)


class ActuatorDriver(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def fire_reject(self, item_id: str, delay_ms: float = 0) -> bool: ...
    def fire_hold(self, item_id: str) -> bool: ...


class RejectController:
    """Orchestrates rejection mechanism.

    Features:
    - Belt speed & distance compensation for timed reject firing.
    - Dry-run mode (logs reject decision without hardware trigger).
    - Fail-safe: if system error or low confidence → divert to HOLD FOR REVIEW (never pass silently).
    - Acknowledgement check & log tracking.
    """

    def __init__(
        self,
        driver: Optional[ActuatorDriver] = None,
        dry_run: bool = False,
        conveyor_distance_mm: float = 500.0,
        belt_speed_mm_s: float = 500.0,
    ):
        self.driver = driver if driver is not None else SimulatedDriver()
        self.dry_run = dry_run
        self.conveyor_distance_mm = conveyor_distance_mm
        self.belt_speed_mm_s = belt_speed_mm_s
        self.total_triggers = 0
        self.successful_acks = 0

    def start(self):
        self.driver.connect()

    def stop(self):
        self.driver.disconnect()

    def calculate_reject_delay_ms(self, scan_timestamp_ms: float) -> float:
        """Calculate millisecond delay until item reaches reject mechanism based on belt speed."""
        if self.belt_speed_mm_s <= 0:
            return 0.0
        travel_time_ms = (self.conveyor_distance_mm / self.belt_speed_mm_s) * 1000.0
        elapsed_ms = (time.time() * 1000.0) - scan_timestamp_ms
        remaining_delay = max(0.0, travel_time_ms - elapsed_ms)
        return remaining_delay

    def process_verdict(
        self,
        item_id: str,
        verdict: str,
        scan_timestamp_ms: float,
        system_healthy: bool = True,
    ) -> tuple[str, bool]:
        """Process verdict and execute actuation.

        Args:
            item_id: Unique scan item ID.
            verdict: PASS | REJECT | REVIEW.
            scan_timestamp_ms: Timestamp when item was scanned.
            system_healthy: If False, fail-safe activates -> divert to HOLD.

        Returns:
            Tuple of (actuator_action, actuator_ack).
        """
        self.total_triggers += 1

        # FAIL-SAFE: If system is unhealthy or verdict is invalid -> force HOLD/REJECT
        if not system_healthy or verdict not in ("PASS", "REJECT", "REVIEW"):
            logger.warning(f"FAIL-SAFE triggered for item {item_id}! Diverting to HOLD for manual review.")
            if self.dry_run:
                return "hold_dry_run", True
            ack = self.driver.fire_hold(item_id)
            if ack:
                self.successful_acks += 1
            return "hold", ack

        if verdict == "PASS":
            return "pass", True

        delay_ms = self.calculate_reject_delay_ms(scan_timestamp_ms)

        if verdict == "REJECT":
            if self.dry_run:
                logger.info(f"[DRY-RUN] REJECT item {item_id} (computed delay: {delay_ms:.1f}ms)")
                return "reject_dry_run", True

            ack = self.driver.fire_reject(item_id, delay_ms=delay_ms)
            if ack:
                self.successful_acks += 1
            else:
                logger.error(f"Actuator failed to acknowledge REJECT for item {item_id}!")
            return "reject", ack

        if verdict == "REVIEW":
            if self.dry_run:
                logger.info(f"[DRY-RUN] HOLD item {item_id}")
                return "hold_dry_run", True

            ack = self.driver.fire_hold(item_id)
            if ack:
                self.successful_acks += 1
            return "hold", ack

        return "pass", True
