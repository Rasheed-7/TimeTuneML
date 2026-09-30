"""Bayesian hyperparameter optimization for TimeTuneML using Optuna TPE.

Milestone 3: standard TPE-based Bayesian optimization.
Milestone 4: optional global time budget with cumulative cost tracking.
Milestone 5: optional time-aware (cost-aware) trial selection layer.

Design notes
------------
* Reuses the leakage-safe preprocessing pipeline and metric utilities
  from Milestones 1 and 2.
* The train/test split is created ONCE and reused for every trial.
* If ``budget_seconds`` is provided, the optimizer stops as soon as the
  budget is exhausted (checked *before* each new trial).
* If ``cost_aware=True``, the optimizer uses Optuna's ask/tell API so it
  can inspect each proposed candidate before committing to evaluation.
  During a warm-up window it behaves exactly like standard TPE; after
  that, a lightweight execution-cost model predicts how long the
  candidate is likely to take, and candidates whose predicted cost
  exceeds the remaining budget are rejected (tell-ed as failed) so a
  fresh candidate is proposed. This is a project-level *time-aware*
  strategy layered on top of standard Bayesian/TPE search — not a novel
  Bayesian optimization algorithm.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from timetuneml.evaluation.metrics import compute_classification_metrics
from timetuneml.models.registry import RANDOM_STATE
from timetuneml.optimization.base import BaseOptimizer, TrialRecord
from timetuneml.optimization.budget import BudgetController
from timetuneml.optimization.cost_aware import (
    ExecutionCostModel,
    cost_aware_utility,
)
from timetuneml.preprocessing.pipeline import build_preprocessing_pipeline

PRIMARY_METRIC = "f1"
SAFETY_MAX_TRIALS = 1000

# When cost-aware, how many candidate proposals to try per trial before
# giving up and evaluating whatever Optuna proposes next. Prevents an
# unbounded rejection loop.
MAX_CANDIDATE_REJECTIONS = 10

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianOptimizationOptimizer(BaseOptimizer):
    """Bayesian optimization over a search space using Optuna's TPE sampler.

    See module docstring for milestone history.
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
        # --- Milestone 5 ---
        cost_aware: bool = False,
        warmup_trials: int = 5,
        cost_model_alpha: float = 1.0,
        min_remaining_budget_seconds: float = 0.0,
    ) -> None:
        if n_trials <= 0:
            raise ValueError("n_trials must be a positive integer.")
        if budget_seconds is not None and budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative.")
        if warmup_trials < 0:
            raise ValueError("warmup_trials must be non-negative.")

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

        # Milestone 5 knobs
        self.cost_aware = bool(cost_aware)
        self.warmup_trials = int(warmup_trials)
        self.cost_model = ExecutionCostModel(alpha=cost_model_alpha)
        self.min_remaining_budget_seconds = float(min_remaining_budget_seconds)

        # State
        self.trials: List[TrialRecord] = []
        self.total_duration_seconds: float = 0.0
        self.budget_exhausted: bool = False
        self._study: Optional[optuna.Study] = None

        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    @staticmethod
    def _default_model_factory(**params: Any) -> RandomForestClassifier:
        # n_jobs=1 is deliberate. The optimizer already runs trials
        # sequentially, and each trial's RF fit is tiny on these datasets.
        # Using n_jobs=-1 inside an outer search loop causes joblib to
        # spawn and tear down a thread pool on every trial, which at
        # thousands of trials per budget collapses throughput and floods
        # the log with sklearn.utils.parallel.delayed warnings.
        # Single-threaded fits keep the loop deterministic and fast enough.
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self) -> List[TrialRecord]:
        """Run the study, honouring budget and (optionally) cost-aware selection."""
        start_all = time.perf_counter()

        (
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)

        self.trials = []
        self.budget_exhausted = False

        budget = (
            BudgetController(self.budget_seconds)
            if self.budget_seconds is not None
            else None
        )
        if budget is not None:
            budget.start()

        # ------------------------------------------------------------------
        # Trial loop
        # ------------------------------------------------------------------
        completed_trials = 0
        while True:
            # --- stop conditions -----------------------------------------
            if budget is not None and budget.is_exhausted():
                self.budget_exhausted = True
                break
            if budget is None and completed_trials >= self.n_trials:
                break
            if budget is not None and completed_trials >= SAFETY_MAX_TRIALS:
                break

            # Refuse to start a new trial when the remaining budget is
            # below the configured floor. (Floor defaults to 0, so this
            # is a no-op unless the caller opts in.)
            if (
                budget is not None
                and budget.remaining() <= self.min_remaining_budget_seconds
            ):
                self.budget_exhausted = True
                break

            # --- choose the trial path -----------------------------------
            use_cost_aware_now = (
                self.cost_aware and completed_trials >= self.warmup_trials
            )

            if use_cost_aware_now:
                record = self._run_cost_aware_trial(
                    trial_id=completed_trials + 1,
                    budget=budget,
                )
            else:
                record = self._run_standard_trial(
                    trial_id=completed_trials + 1,
                    budget=budget,
                    start_all=start_all,
                )

            self.trials.append(record)
            completed_trials += 1

            # --- update cost model with the newly observed real cost -----
            if (
                record.status == "COMPLETE"
                and record.duration_seconds > 0.0
            ):
                try:
                    history_params = [t.parameters for t in self.trials
                                      if t.status == "COMPLETE"]
                    history_durations = [t.duration_seconds for t in self.trials
                                         if t.status == "COMPLETE"]
                    self.cost_model.fit(history_params, history_durations)
                except Exception:
                    # Fitting must never crash the search.
                    pass

        self.total_duration_seconds = time.perf_counter() - start_all
        return self.trials

    def best_trial(self) -> Optional[TrialRecord]:
        """Return the completed trial with the highest primary metric."""
        completed = [
            t for t in self.trials
            if t.status == "COMPLETE" and t.score is not None
        ]
        if not completed:
            return None
        return max(completed, key=lambda t: t.score)  # type: ignore[arg-type]

    def best_utility_trial(self) -> Optional[TrialRecord]:
        """Return the trial with the highest recorded utility, if any."""
        scored = [t for t in self.trials if t.utility is not None]
        if not scored:
            return None
        return max(scored, key=lambda t: t.utility)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Standard (warm-up / non-cost-aware) trial
    # ------------------------------------------------------------------
    def _run_standard_trial(
        self,
        trial_id: int,
        budget: Optional[BudgetController],
        start_all: float,
    ) -> TrialRecord:
        """Run one trial using Optuna's normal optimize() path."""
        self._study.optimize(
            self._objective,
            n_trials=1,
            catch=(Exception,),
        )
        optuna_trial = self._study.trials[-1]
        cumulative = (
            budget.elapsed() if budget is not None
            else time.perf_counter() - start_all
        )
        return self._record_from_optuna_trial(
            optuna_trial=optuna_trial,
            trial_id=trial_id,
            cumulative_time_seconds=cumulative,
        )

    # ------------------------------------------------------------------
    # Cost-aware trial
    # ------------------------------------------------------------------
    def _run_cost_aware_trial(
        self,
        trial_id: int,
        budget: Optional[BudgetController],
    ) -> TrialRecord:
        """Cost-aware path: ask for a candidate, then decide whether to run it.

        Uses Optuna's ask/tell so we can:
          1. Inspect the candidate's predicted cost before evaluating it.
          2. Reject candidates whose predicted cost exceeds the remaining
             budget by telling Optuna the trial failed, then asking for a
             fresh candidate.

        If estimation or rejection goes wrong, we fall back to evaluating
        whatever candidate Optuna last proposed.
        """
        chosen_trial: Optional[optuna.Trial] = None
        rejected = 0
        last_predicted_cost: Optional[float] = None

        while rejected < MAX_CANDIDATE_REJECTIONS:
            trial = self._study.ask()
            params = self._sample_params(trial)

            predicted_cost = None
            if self.cost_model.is_fitted:
                predicted_cost = self.cost_model.predict(params)

            # Rejection rule: reject only when we have a valid estimate
            # AND it clearly exceeds the remaining budget. If we have no
            # estimate, accept the candidate (measured time remains
            # authoritative).
            if (
                predicted_cost is not None
                and budget is not None
                and predicted_cost > budget.remaining()
            ):
                last_predicted_cost = predicted_cost
                rejected += 1
                self._study.tell(
                    trial,
                    state=optuna.trial.TrialState.FAIL,
                )
                continue

            chosen_trial = trial
            last_predicted_cost = predicted_cost
            break

        # If every proposal was rejected, fall back to a normal trial so
        # we don't stop the search on a modelling artefact.
        if chosen_trial is None:
            return self._run_standard_trial(
                trial_id=trial_id, budget=budget, start_all=time.perf_counter()
            )

        # ---- evaluate the chosen candidate ----------------------------
        try:
            start = time.perf_counter()
            metrics = self._evaluate_params(chosen_trial.user_attrs.get(
                "parameters", self._sample_params(chosen_trial)
            ))
            actual_duration = time.perf_counter() - start

            self._study.tell(chosen_trial, metrics[self.primary_metric])

            cumulative = (
                budget.elapsed() if budget is not None
                else actual_duration
            )
            utility = None
            if last_predicted_cost is not None:
                utility = cost_aware_utility(
                    predicted_performance=float(metrics[self.primary_metric]),
                    predicted_cost=float(last_predicted_cost),
                )

            record = TrialRecord(
                trial_id=trial_id,
                parameters=self._extract_parameters(chosen_trial),
                score=float(metrics[self.primary_metric]),
                duration_seconds=float(actual_duration),
                cumulative_time_seconds=float(cumulative),
                status="COMPLETE",
                metrics=metrics,
                estimated_duration_seconds=(
                    float(last_predicted_cost) if last_predicted_cost is not None else None
                ),
                utility=utility,
            )
            return record
        except Exception as exc:  # noqa: BLE001
            try:
                self._study.tell(
                    chosen_trial,
                    state=optuna.trial.TrialState.FAIL,
                )
            except Exception:
                pass
            cumulative = budget.elapsed() if budget is not None else 0.0
            return TrialRecord(
                trial_id=trial_id,
                parameters=self._extract_parameters(chosen_trial),
                score=None,
                duration_seconds=0.0,
                cumulative_time_seconds=float(cumulative),
                status="FAILED",
                metrics={},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Shared evaluation (used by cost-aware path)
    # ------------------------------------------------------------------
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Fit preprocessing + model on train, evaluate on test, return metrics.

        This mirrors the body of ``_objective`` so the cost-aware path
        reuses the same leakage-safe procedure.
        """
        preprocessor = build_preprocessing_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )
        X_train_t = preprocessor.fit_transform(self._X_train, self._y_train)
        X_test_t = preprocessor.transform(self._X_test)

        model = self.model_factory(**params)
        model.fit(X_train_t, self._y_train)

        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)
        return compute_classification_metrics(self._y_test, y_pred, y_proba)

    def _extract_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Recover the human-readable parameter dict from an ask()-ed trial.

        Optuna stores the suggestion under ``params`` with the internal
        key names (including the ``__idx`` suffix for choice params), so
        we normalise back to real values.
        """
        params = dict(trial.params)
        params.update(trial.user_attrs.get("parameters", {}))
        for name, spec in self.search_space.items():
            if spec["type"] == "choice":
                idx_key = f"{name}__idx"
                if idx_key in params:
                    options = list(spec["options"])
                    params[name] = options[int(params.pop(idx_key))]
        return params

    # ------------------------------------------------------------------
    # Optuna objective (used by standard / warm-up trials)
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._sample_params(trial)
        trial.set_user_attr("parameters", params)
        metrics = self._evaluate_params(params)
        trial.set_user_attr("metrics", metrics)
        return float(metrics[self.primary_metric])

    # ------------------------------------------------------------------
    # Sampling (shared)
    # ------------------------------------------------------------------
    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            kind = spec["type"]
            if kind == "int":
                params[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"])
                )
            elif kind == "float":
                params[name] = trial.suggest_float(
                    name, float(spec["low"]), float(spec["high"])
                )
            elif kind == "choice":
                options = list(spec["options"])
                idx = trial.suggest_int(f"{name}__idx", 0, len(options) - 1)
                params[name] = options[idx]
            else:
                raise ValueError(f"Unknown search-space spec type: {kind!r}")
        return params

    # ------------------------------------------------------------------
    # Recording (standard / warm-up path)
    # ------------------------------------------------------------------
    def _record_from_optuna_trial(
        self,
        optuna_trial: optuna.trial.FrozenTrial,
        trial_id: int,
        cumulative_time_seconds: float,
    ) -> TrialRecord:
        duration = float(
            (optuna_trial.datetime_complete - optuna_trial.datetime_start).total_seconds()
        ) if (
            optuna_trial.datetime_complete is not None
            and optuna_trial.datetime_start is not None
        ) else 0.0

        params = dict(optuna_trial.params)
        params.update(optuna_trial.user_attrs.get("parameters", {}))

        for name, spec in self.search_space.items():
            if spec["type"] == "choice":
                idx_key = f"{name}__idx"
                if idx_key in params:
                    options = list(spec["options"])
                    params[name] = options[int(params.pop(idx_key))]

        if optuna_trial.state == optuna.trial.TrialState.COMPLETE:
            metrics = optuna_trial.user_attrs.get("metrics", {})
            return TrialRecord(
                trial_id=trial_id,
                parameters=params,
                score=float(optuna_trial.value) if optuna_trial.value is not None else None,
                duration_seconds=duration,
                cumulative_time_seconds=float(cumulative_time_seconds),
                status="COMPLETE",
                metrics=metrics,
            )

        error_msg = None
        if optuna_trial.state == optuna.trial.TrialState.FAIL:
            fail_exc = getattr(optuna_trial, "_exception", None)
            error_msg = repr(fail_exc) if fail_exc is not None else "Trial failed."

        return TrialRecord(
            trial_id=trial_id,
            parameters=params,
            score=None,
            duration_seconds=duration,
            cumulative_time_seconds=float(cumulative_time_seconds),
            status="FAILED",
            metrics={},
            error=error_msg,
        )