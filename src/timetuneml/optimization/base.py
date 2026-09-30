"""Base types shared by all TimeTuneML optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrialRecord:
    """Result of a single optimization trial."""

    trial_id: int
    parameters: Dict[str, Any]
    score: Optional[float]
    duration_seconds: float
    status: str = "COMPLETE"
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    # Milestone 4: cumulative time (seconds) since the search started.
    cumulative_time_seconds: float = 0.0
    # Milestone 5: cost-aware diagnostics. None during warm-up or when
    # the cost model could not produce a valid prediction.
    estimated_duration_seconds: Optional[float] = None
    utility: Optional[float] = None


class BaseOptimizer:
    """Minimal interface every TimeTuneML optimizer should follow."""

    def optimize(self):
        raise NotImplementedError

    def best_trial(self):
        raise NotImplementedError