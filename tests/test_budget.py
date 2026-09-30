"""Tests for the TimeTuneML BudgetController.

The controller is a monotonic-time wrapper used by the optimizers to
enforce a global time budget. Tests are fast and do not sleep longer
than a few milliseconds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timetuneml.optimization.budget import BudgetController  # noqa: E402


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        BudgetController(budget_seconds=-1.0)


def test_elapsed_is_zero_before_start():
    bc = BudgetController(budget_seconds=10.0)
    assert bc.elapsed() == 0.0
    assert bc.remaining() == 10.0
    assert not bc.is_exhausted()


def test_started_property_reflects_lifecycle():
    bc = BudgetController(budget_seconds=10.0)
    assert bc.started is False
    bc.start()
    assert bc.started is True


def test_elapsed_increases_after_start():
    bc = BudgetController(budget_seconds=10.0)
    bc.start()
    t1 = bc.elapsed()
    time.sleep(0.005)  # 5 ms
    t2 = bc.elapsed()
    assert t2 > t1


def test_remaining_decreases_monotonically():
    bc = BudgetController(budget_seconds=10.0)
    bc.start()
    r1 = bc.remaining()
    time.sleep(0.005)
    r2 = bc.remaining()
    assert r2 <= r1


def test_zero_budget_is_immediately_exhausted_after_start():
    bc = BudgetController(budget_seconds=0.0)
    bc.start()
    assert bc.is_exhausted()
    assert bc.remaining() == 0.0


def test_tiny_budget_is_exhausted_after_short_sleep():
    bc = BudgetController(budget_seconds=0.005)
    bc.start()
    time.sleep(0.010)
    assert bc.is_exhausted()


def test_remaining_never_goes_negative():
    bc = BudgetController(budget_seconds=0.005)
    bc.start()
    time.sleep(0.010)
    assert bc.remaining() == 0.0