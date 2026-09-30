"""Random Search optimizer for TimeTuneML.

Milestone 2: standard Random Search over a search space.
Milestone 6: optional global time budget so the benchmark can compare
Random Search against the Bayesian optimizers under the same budget.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from timetuneml.evaluation.metrics import compute_classification_metrics
from timetuneml.models.registry import RANDOM_STATE
from timetuneml.optimization.base import BaseOptimizer, TrialRecord
from timetuneml.optimization.budget import BudgetController
from timetuneml.preprocessing.pipeline import build_preprocessing_pipeline

PRIMARY_METRIC = "f1"
SAFETY_MAX_TRIALS = 1000


def _sample_parameters(
    search_space: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Draw one value per hyperparameter according to its spec."""
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif kind == "float":
            params[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif kind == "choice":
            options = spec["options"]
            idx = int(rng.integers(0, len(options)))
            params[name] = options[idx]
        else:
            raise ValueError(f"Unknown search-space spec type: {kind!r}")
    return params


class RandomSearchOptimizer(BaseOptimizer):
    """Random Search over a hyperparameter search space.

    Parameters
    ----------
    X, y : np.ndarray
        Full dataset. Train/test split is created internally.
    numeric_features, categorical_features : sequences of str
        Column identifiers used by the preprocessing pipeline.
    search_space : dict
        Search space from ``models.search_spaces``.
    n_trials : int
        Number of trials to run when no budget is set.
    random_state : int
        Seed for sampling and model construction. Defaults to 42.
    test_size : float
        Fraction of data used for evaluation. Defaults to 0.2.
    model_factory : callable, optional
        Function taking ``**params`` returning an sklearn estimator.
    primary_metric : str
        Metric used to pick the best trial. Defaults to "f1".
    budget_seconds : float, optional
        When provided, the optimizer runs until this global monotonic
        budget is exhausted. When ``None`` (default), it stops after
        ``n_trials`` — the Milestone 2 behaviour.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        numeric_features: Sequence[str],
        categorical_features: Optional[Sequence[str]],
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int = 10,
        random_state: int = RANDOM_STATE,
        test_size: float = 0.2,
        model_factory: Optional[Callable[..., Any]] = None,
        primary_metric: str = PRIMARY_METRIC,
        budget_seconds: Optional[float] = None,
    ) -> None:
        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")
        if budget_seconds is not None and budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative.")

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self.search_space = search_space
        self.n_trials = int(n_trials)
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.primary_metric = primary_metric
        self.model_factory = model_factory or self._default_model_factory
        self.budget_seconds = (
            float(budget_seconds) if budget_seconds is not None else None
        )

        self.trials: List[TrialRecord] = []
        self.total_duration_seconds: float = 0.0
        self.budget_exhausted: bool = False

    @staticmethod
    def _default_model_factory(**params: Any) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)

    def optimize(self) -> List[TrialRecord]:
        start_all = time.perf_counter()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        rng = np.random.default_rng(self.random_state)
        self.trials = []
        self.budget_exhausted = False

        budget = (
            BudgetController(self.budget_seconds)
            if self.budget_seconds is not None
            else None
        )
        if budget is not None:
            budget.start()

        trial_id = 0
        while True:
            if budget is not None and budget.is_exhausted():
                self.budget_exhausted = True
                break
            if budget is None and trial_id >= self.n_trials:
                break
            if budget is not None and trial_id >= SAFETY_MAX_TRIALS:
                break

            trial_id += 1
            params = _sample_parameters(self.search_space, rng)
            record = self._run_single_trial(
                trial_id=trial_id,
                params=params,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                budget=budget,
                start_all=start_all,
            )
            self.trials.append(record)

        self.total_duration_seconds = time.perf_counter() - start_all
        return self.trials

    def best_trial(self) -> Optional[TrialRecord]:
        completed = [
            t for t in self.trials
            if t.status == "COMPLETE" and t.score is not None
        ]
        if not completed:
            return None
        return max(completed, key=lambda t: t.score)  # type: ignore[arg-type]

    def _run_single_trial(
        self,
        trial_id: int,
        params: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        budget: Optional[BudgetController],
        start_all: float,
    ) -> TrialRecord:
        start = time.perf_counter()
        try:
            preprocessor = build_preprocessing_pipeline(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
            )
            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_test_t = preprocessor.transform(X_test)

            model = self.model_factory(**params)
            model.fit(X_train_t, y_train)

            y_pred = model.predict(X_test_t)
            y_proba = model.predict_proba(X_test_t)

            metrics = compute_classification_metrics(y_test, y_pred, y_proba)
            duration = time.perf_counter() - start
            cumulative = (
                budget.elapsed() if budget is not None
                else time.perf_counter() - start_all
            )

            return TrialRecord(
                trial_id=trial_id,
                parameters=dict(params),
                score=float(metrics[self.primary_metric]),
                duration_seconds=float(duration),
                cumulative_time_seconds=float(cumulative),
                status="COMPLETE",
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start
            cumulative = (
                budget.elapsed() if budget is not None
                else time.perf_counter() - start_all
            )
            return TrialRecord(
                trial_id=trial_id,
                parameters=dict(params),
                score=None,
                duration_seconds=float(duration),
                cumulative_time_seconds=float(cumulative),
                status="FAILED",
                metrics={},
                error=str(exc),
            )