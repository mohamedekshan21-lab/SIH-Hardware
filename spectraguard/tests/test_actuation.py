"""Unit tests for actuation controller delay calculation and fail-safe logic."""

import time
import pytest
from backend.actuation.controller import RejectController
from backend.actuation.simulated_driver import SimulatedDriver


def test_reject_delay_calculation():
    # Distance = 500mm, Speed = 500mm/s -> Travel time = 1000ms
    controller = RejectController(conveyor_distance_mm=500.0, belt_speed_mm_s=500.0)
    now_ms = time.time() * 1000.0

    delay = controller.calculate_reject_delay_ms(now_ms)
    assert 950.0 <= delay <= 1050.0


def test_actuation_verdict_pass():
    controller = RejectController(dry_run=True)
    action, ack = controller.process_verdict("ITEM-1", "PASS", time.time() * 1000.0)
    assert action == "pass"
    assert ack is True


def test_actuation_verdict_reject():
    controller = RejectController(dry_run=True)
    action, ack = controller.process_verdict("ITEM-2", "REJECT", time.time() * 1000.0)
    assert action == "reject_dry_run"
    assert ack is True


def test_actuation_failsafe_unhealthy():
    controller = RejectController(dry_run=False)
    # When system_healthy is False, verdict must divert to HOLD for manual review
    action, ack = controller.process_verdict("ITEM-3", "PASS", time.time() * 1000.0, system_healthy=False)
    assert action == "hold"
    assert ack is True
