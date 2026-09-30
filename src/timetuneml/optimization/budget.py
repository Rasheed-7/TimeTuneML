"""Global time-budget controller for TimeTuneML.

Provides a small, reusable helper that tracks a monotonic elapsed time
against a fixed budget in seconds. Used by budget-aware optimizers to
decide whether another trial may start.

Design notes
------------
* Uses ``time.perf_counter()`` — a monotonic clock — so elapsed time is
  unaffected by wall-clock adjustments (NTP, DST, manual clock changes).
* The controller does not schedule anything; it only answers "how much
  time has elapsed?" and "is the budget exhausted?".
"""

from __future__ import annotations

import time


class BudgetController:
    """Track a fixed time budget using a monotonic clock.

    Parameters
    ----------
    budget_seconds : float
        Total wall-clock (monotonic) time allowed for the run. Must be
        non-negative.
    """

    def __init__(self, budget_seconds: float) -> None:
        if budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative.")
        self.budget_seconds = float(budget_seconds)
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start (or restart) the budget timer."""
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        """Return the elapsed time in seconds since ``start()``.

        If ``start()`` has not been called yet, returns 0.0.
        """
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def remaining(self) -> float:
        """Return remaining seconds. Never negative."""
        return max(0.0, self.budget_seconds - self.elapsed())

    def is_exhausted(self) -> bool:
        """Return True if the elapsed time has reached the budget."""
        return self.elapsed() >= self.budget_seconds

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._start_time is not None