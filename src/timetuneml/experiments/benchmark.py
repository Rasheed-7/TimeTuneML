"""Benchmarking framework for TimeTuneML.

Orchestrates four methods under the same dataset, model, preprocessing,
search space, metric, seed, and budget:

    1. Default Random Forest (Milestone 1 baseline).
    2. Random Search (Milestone 2).
    3. Bayesian Optimization (Milestone 3/4 — Optuna TPE).
    4. Time-Aware Bayesian Optimization (Milestone 5 — cost-aware).

This module is responsible for orchestration only. It does not
reimplement dataset loading, preprocessing, model construction, metric
calculation, or search spaces — those are reused from the existing
Milestone 1–5 modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from timetuneml.data.loader import load_breast_cancer_dataset
from timetuneml.data.validation import validate_dataset
from timetuneml.evaluation.metrics import compute_classification_metrics
from timetuneml.models.registry import RANDOM_STATE, get_baseline_random_forest
from timetuneml.models.search_spaces import get_random_forest_search_space
from timetuneml.optimization.bayesian import BayesianOptimizationOptimizer
from timetuneml.optimization.random_search import RandomSearchOptimizer
from timetuneml.preprocessing.pipeline import build_preprocessing_pipeline

# Method identifiers used throughout the benchmark and CSV output.
METHOD_DEFAULT = "Default RF"
METHOD_RANDOM = "Random Search"
METHOD_BAYESIAN = "Bayesian Optimization"
METHOD_COST_AWARE = "Time-Aware Bayesian"

PRIMARY_METRIC = "f1"


@dataclass
class BenchmarkResult:
    """Normalised record for one method's benchmark run."""

    method: str
    seed: int
    budget_seconds: float
    best_f1: Optional[float]
    total_time_seconds: float
    num_trials: int
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    best_utility: Optional[float] = None
    average_trial_time: Optional[float] = None

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        # Serialise parameters to a compact string so the CSV stays tidy.
        row["best_parameters"] = (
            ";".join(f"{k}={v}" for k, v in self.best_parameters.items())
            if self.best_parameters else ""
        )
        return row


# ----------------------------------------------------------------------
# Shared data
# ----------------------------------------------------------------------
def _load_shared_dataset():
    """Load + validate the breast cancer dataset (Milestone 1 path)."""
    X, y, feature_names, _target_names = load_breast_cancer_dataset()
    validate_dataset(X, y)
    return X, y, feature_names


# ----------------------------------------------------------------------
# Method 1 — Default Random Forest
# ----------------------------------------------------------------------
def run_default_rf(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features,
    seed: int = RANDOM_STATE,
    budget_seconds: float = 30.0,
    test_size: float = 0.2,
) -> BenchmarkResult:
    """Evaluate the default Random Forest baseline.

    Uses the same train/test split configuration as the search methods.
    ``num_trials`` is always 1 for the default baseline (there is no
    search). ``budget_seconds`` is recorded for reporting only — the
    baseline is not budget-limited because there is nothing to search.
    """
    import time

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features, categorical_features=None,
    )

    start = time.perf_counter()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    model = get_baseline_random_forest(random_state=seed)
    model.fit(X_train_t, y_train)

    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    total_time = time.perf_counter() - start

    return BenchmarkResult(
        method=METHOD_DEFAULT,
        seed=seed,
        budget_seconds=budget_seconds,
        best_f1=float(metrics[PRIMARY_METRIC]),
        total_time_seconds=float(total_time),
        num_trials=1,
        best_parameters=dict(model.get_params()),
    )


# ----------------------------------------------------------------------
# Method 2 — Random Search
# ----------------------------------------------------------------------
def run_random_search(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features,
    seed: int = RANDOM_STATE,
    budget_seconds: float = 30.0,
    test_size: float = 0.2,
) -> BenchmarkResult:
    optimizer = RandomSearchOptimizer(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        test_size=test_size,
        budget_seconds=budget_seconds,
    )
    optimizer.optimize()
    best = optimizer.best_trial()

    return BenchmarkResult(
        method=METHOD_RANDOM,
        seed=seed,
        budget_seconds=budget_seconds,
        best_f1=best.score if best is not None else None,
        total_time_seconds=float(optimizer.total_duration_seconds),
        num_trials=len(optimizer.trials),
        best_parameters=best.parameters if best is not None else {},
    )


# ----------------------------------------------------------------------
# Method 3 — Bayesian Optimization
# ----------------------------------------------------------------------
def run_bayesian(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features,
    seed: int = RANDOM_STATE,
    budget_seconds: float = 30.0,
    test_size: float = 0.2,
) -> BenchmarkResult:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        test_size=test_size,
        budget_seconds=budget_seconds,
        cost_aware=False,
    )
    optimizer.optimize()
    best = optimizer.best_trial()

    return BenchmarkResult(
        method=METHOD_BAYESIAN,
        seed=seed,
        budget_seconds=budget_seconds,
        best_f1=best.score if best is not None else None,
        total_time_seconds=float(optimizer.total_duration_seconds),
        num_trials=len(optimizer.trials),
        best_parameters=best.parameters if best is not None else {},
    )


# ----------------------------------------------------------------------
# Method 4 — Time-Aware Bayesian
# ----------------------------------------------------------------------
def run_cost_aware(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features,
    seed: int = RANDOM_STATE,
    budget_seconds: float = 30.0,
    test_size: float = 0.2,
    warmup_trials: int = 5,
) -> BenchmarkResult:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        test_size=test_size,
        budget_seconds=budget_seconds,
        cost_aware=True,
        warmup_trials=warmup_trials,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    best_util_trial = optimizer.best_utility_trial()

    completed = [t for t in optimizer.trials if t.status == "COMPLETE"]
    avg_trial_time = (
        sum(t.duration_seconds for t in completed) / len(completed)
        if completed else None
    )

    return BenchmarkResult(
        method=METHOD_COST_AWARE,
        seed=seed,
        budget_seconds=budget_seconds,
        best_f1=best.score if best is not None else None,
        total_time_seconds=float(optimizer.total_duration_seconds),
        num_trials=len(optimizer.trials),
        best_parameters=best.parameters if best is not None else {},
        best_utility=(
            float(best_util_trial.utility)
            if best_util_trial is not None and best_util_trial.utility is not None
            else None
        ),
        average_trial_time=(
            float(avg_trial_time) if avg_trial_time is not None else None
        ),
    )


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------
def run_all(
    seed: int = RANDOM_STATE,
    budget_seconds: float = 30.0,
    test_size: float = 0.2,
    warmup_trials: int = 5,
) -> list[BenchmarkResult]:
    """Run all four methods under the same configuration and return results."""
    X, y, feature_names = _load_shared_dataset()

    results: list[BenchmarkResult] = []
    results.append(
        run_default_rf(
            X, y, feature_names,
            seed=seed, budget_seconds=budget_seconds, test_size=test_size,
        )
    )
    results.append(
        run_random_search(
            X, y, feature_names,
            seed=seed, budget_seconds=budget_seconds, test_size=test_size,
        )
    )
    results.append(
        run_bayesian(
            X, y, feature_names,
            seed=seed, budget_seconds=budget_seconds, test_size=test_size,
        )
    )
    results.append(
        run_cost_aware(
            X, y, feature_names,
            seed=seed, budget_seconds=budget_seconds, test_size=test_size,
            warmup_trials=warmup_trials,
        )
    )
    return results