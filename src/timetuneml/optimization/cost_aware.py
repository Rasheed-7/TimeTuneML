"""Cost-aware utilities for TimeTuneML.

This module provides two small, self-contained building blocks that
together implement a *project-level* time-aware optimization strategy on
top of standard Optuna/TPE:

  1. ``ExecutionCostModel`` — a lightweight regression model that
     predicts how long a candidate Random Forest configuration will take
     to evaluate, based on historical trial durations.
  2. ``cost_aware_utility`` — a scalar utility that combines predicted
     performance and predicted cost, so that trial selection can favour
     configurations that are efficient *as well as* accurate.

Design notes
------------
* The cost model predicts **computational cost only**. It never sees the
  test-set F1 and is never used as a performance surrogate.
* The cost model is deliberately small (Ridge regression on a handful of
  hyperparameters) — the goal at this milestone is transparency, not
  state-of-the-art regression.
* If the model cannot be fitted, or produces invalid predictions, the
  caller falls back to standard TPE selection. Nothing here raises on
  the normal "not enough data yet" path.
* Nothing in this module replaces Optuna. Optuna/TPE still proposes the
  candidate hyperparameters; this layer only decides whether a proposed
  candidate looks worth evaluating under the remaining time budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge

# Small epsilon to avoid division-by-zero in the utility computation.
# Chosen well below typical Random Forest durations (~0.05–1.0 s) so that
# it only matters when the cost estimate collapses to zero.
EPSILON_COST = 1e-3

# Number of observed trials required before the cost model is considered
# usable. Below this, the caller should stay in warm-up mode.
MIN_OBSERVATIONS_FOR_COST_MODEL = 3


def _encode_params(params: Dict[str, Any], feature_order: Sequence[str]) -> List[float]:
    """Convert one parameter dict into a numeric feature vector.

    ``max_features`` is the only non-numeric field in the search space;
    we encode it as a small integer index so the model can still learn a
    cost signal from it.
    """
    max_features_choices = ["sqrt", "log2", 0.3, 0.5, 1.0]
    row: List[float] = []
    for name in feature_order:
        value = params.get(name)
        if name == "max_features":
            # Map categorical choice to an ordinal index; unknown values
            # fall back to -1 so the model treats them as "other".
            try:
                row.append(float(max_features_choices.index(value)))
            except ValueError:
                row.append(-1.0)
        else:
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                row.append(0.0)
    return row


class ExecutionCostModel:
    """Predict execution duration from Random Forest hyperparameters.

    Uses a Ridge regression on numeric-encoded hyperparameters. This is
    intentionally simple: the purpose at this milestone is to demonstrate
    a measurable cost-aware selection layer, not to build a highly
    accurate cost surrogate.
    """

    # Feature order used by the model. Kept explicit so encoding is
    # deterministic and inspectable.
    FEATURE_ORDER: Sequence[str] = (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    )

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._model: Optional[Ridge] = None
        self._n_observations: int = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def n_observations(self) -> int:
        return self._n_observations

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def fit(
        self,
        parameter_history: Sequence[Dict[str, Any]],
        duration_history: Sequence[float],
    ) -> bool:
        """Fit the model on observed (params, duration) pairs.

        Returns True on success, False if the model could not be fitted
        (insufficient data, degenerate targets, etc.). Never raises for
        ordinary data-shortage reasons — the caller decides what to do.
        """
        if len(parameter_history) != len(duration_history):
            return False
        if len(parameter_history) < MIN_OBSERVATIONS_FOR_COST_MODEL:
            return False

        try:
            X = np.asarray(
                [_encode_params(p, self.FEATURE_ORDER) for p in parameter_history],
                dtype=float,
            )
            y = np.asarray(duration_history, dtype=float)

            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
                return False
            if np.allclose(y, y[0]):
                # Constant target — nothing to learn, predictions would be
                # a trivially valid but uninformative constant.
                return False

            model = Ridge(alpha=self.alpha)
            model.fit(X, y)
            self._model = model
            self._n_observations = len(y)
            return True
        except Exception:
            # Fitting must never crash the search. Caller falls back to
            # standard TPE selection.
            self._model = None
            self._n_observations = 0
            return False

    def predict(self, params: Dict[str, Any]) -> Optional[float]:
        """Predict duration for one configuration.

        Returns a positive float, or ``None`` if the model is not fitted
        or the prediction is invalid (non-finite, non-positive).
        """
        if self._model is None:
            return None
        try:
            x = np.asarray(
                [_encode_params(params, self.FEATURE_ORDER)], dtype=float
            )
            pred = float(self._model.predict(x)[0])
        except Exception:
            return None

        if not np.isfinite(pred) or pred <= 0.0:
            return None
        return pred


def cost_aware_utility(
    predicted_performance: float,
    predicted_cost: float,
    epsilon: float = EPSILON_COST,
) -> float:
    """Combine predicted performance and predicted cost into a scalar.

    utility = predicted_performance / max(predicted_cost, epsilon)

    A higher utility means "more performance per second spent". The
    epsilon guards against division by zero when a cost prediction is
    extremely small or (through numerical noise) zero.

    Note: this is a plain scalarisation, not a novel acquisition
    function. It is best described as a project-level heuristic layered
    on top of standard Bayesian/TPE search.
    """
    denom = max(float(predicted_cost), float(epsilon))
    return float(predicted_performance) / denom