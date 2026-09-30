"""Simulated diverter — logs reject actions without real hardware."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class SimulatedDriver:
    """Simulates a pneumatic diverter / reject gate.

    Logs all actions and simulates acknowledgement with configurable
    success rate (to test failure handling).
    """

    def __init__(self, success_rate: float = 0.98, actuation_delay_ms: float = 50.0):
        self.success_rate = success_rate
        self.actuation_delay_ms = actuation_delay_ms
        self._connected = False
        self._fire_count = 0

    def connect(self):
        self._connected = True
        logger.info("SimulatedDriver: connected (demo mode)")

    def disconnect(self):
        self._connected = False
        logger.info("SimulatedDriver: disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def fire_reject(self, item_id: str, delay_ms: float = 0) -> bool:
        """Simulate firing the reject actuator.

        Args:
            item_id: Item being rejected.
            delay_ms: Delay before firing (belt travel time).

        Returns:
            True if the diverter acknowledged successfully.
        """
        import random
        self._fire_count += 1

        # Simulate the actuation delay
        total_delay = (delay_ms + self.actuation_delay_ms) / 1000.0
        time.sleep(min(total_delay, 0.05))  # Cap for demo speed

        success = random.random() < self.success_rate
        status = "ACK" if success else "NACK"
        logger.info(f"SimulatedDriver: REJECT {item_id} → {status} (fire #{self._fire_count})")
        return success

    def fire_hold(self, item_id: str) -> bool:
        """Simulate diverting to 'hold for review' lane."""
        self._fire_count += 1
        logger.info(f"SimulatedDriver: HOLD {item_id} → ACK (fire #{self._fire_count})")
        return True

    @property
    def fire_count(self) -> int:
        return self._fire_count
