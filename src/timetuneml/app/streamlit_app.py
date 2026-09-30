"""Streamlit dashboard for TimeTuneML.

A thin UI layer on top of the existing TimeTuneML components. Every
"Run Optimization" action delegates to one of the optimizers that
already exist in ``timetuneml.optimization`` — this module does not
reimplement any optimization logic.

Run with:
    streamlit run src/timetuneml/app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Make `src/` importable when this file is executed via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timetuneml.data.loader import load_breast_cancer_dataset
from timetuneml.data.validation import validate_dataset
from timetuneml.evaluation.evaluator import run_baseline_experiment
from timetuneml.models.search_spaces import get_random_forest_search_space
from timetuneml.optimization.base import TrialRecord
from timetuneml.optimization.bayesian import BayesianOptimizationOptimizer
from timetuneml.optimization.random_search import RandomSearchOptimizer

# ----------------------------------------------------------------------
# Method identifiers (kept in sync with the sidebar dropdown).
# ----------------------------------------------------------------------
METHOD_DEFAULT = "Default Baseline"
METHOD_RANDOM = "Random Search"
METHOD_BAYESIAN = "Bayesian Optimization"
METHOD_COST_AWARE = "Time-Aware Bayesian Optimization"

ALL_METHODS = [
    METHOD_DEFAULT,
    METHOD_RANDOM,
    METHOD_BAYESIAN,
    METHOD_COST_AWARE,
]

WARMUP_TRIALS = 5


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _trials_to_dataframe(trials: List[TrialRecord], cost_aware: bool) -> pd.DataFrame:
    """Build a DataFrame from real TrialRecord objects.

    Only columns whose data actually exists are populated; for example,
    ``Estimated Duration`` and ``Utility`` appear only when the optimizer
    populated them (cost-aware runs after warm-up).
    """
    rows = []
    for t in trials:
        row = {
            "Trial": t.trial_id,
            "F1": t.score,
            "Duration (s)": t.duration_seconds,
            "Cumulative Time (s)": t.cumulative_time_seconds,
            "Status": t.status,
        }
        if cost_aware:
            row["Estimated Duration (s)"] = t.estimated_duration_seconds
            row["Utility"] = t.utility
        rows.append(row)
    return pd.DataFrame(rows)


def _convergence_dataframe(trials: List[TrialRecord]) -> pd.DataFrame:
    """Return best-so-far F1 per trial, based on completed trials only."""
    rows = []
    best_so_far: Optional[float] = None
    for t in trials:
        if t.status == "COMPLETE" and t.score is not None:
            best_so_far = t.score if best_so_far is None else max(best_so_far, t.score)
            rows.append({"Trial": t.trial_id, "Best F1 so far": best_so_far})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Runners — delegate to the existing optimizers
# ----------------------------------------------------------------------
def _run_default_baseline(
    X, y, feature_names, seed: int, budget_seconds: float
) -> Dict[str, Any]:
    """Run the Milestone 1 baseline. Budget is shown for context only."""
    result = run_baseline_experiment(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        random_state=seed,
    )
    metrics = result["metrics"]
    return {
        "best_f1": metrics["f1"],
        "total_time": result["train_duration_seconds"],
        "num_trials": 1,
        "best_parameters": {},  # baseline uses default RF hyperparameters
        "trials": [],
        "is_cost_aware": False,
        "metrics": metrics,
    }


def _run_random_search(
    X, y, feature_names, seed: int, budget_seconds: float
) -> Dict[str, Any]:
    optimizer = RandomSearchOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    return {
        "best_f1": best.score if best is not None else None,
        "total_time": optimizer.total_duration_seconds,
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "trials": optimizer.trials,
        "is_cost_aware": False,
        "metrics": best.metrics if best is not None else {},
    }


def _run_bayesian(
    X, y, feature_names, seed: int, budget_seconds: float
) -> Dict[str, Any]:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
        cost_aware=False,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    return {
        "best_f1": best.score if best is not None else None,
        "total_time": optimizer.total_duration_seconds,
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "trials": optimizer.trials,
        "is_cost_aware": False,
        "metrics": best.metrics if best is not None else {},
    }


def _run_cost_aware(
    X, y, feature_names, seed: int, budget_seconds: float
) -> Dict[str, Any]:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
        cost_aware=True,
        warmup_trials=WARMUP_TRIALS,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    best_util_trial = optimizer.best_utility_trial()
    return {
        "best_f1": best.score if best is not None else None,
        "total_time": optimizer.total_duration_seconds,
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "trials": optimizer.trials,
        "is_cost_aware": True,
        "best_utility": (
            best_util_trial.utility if best_util_trial is not None else None
        ),
        "metrics": best.metrics if best is not None else {},
    }


_RUNNERS = {
    METHOD_DEFAULT: _run_default_baseline,
    METHOD_RANDOM: _run_random_search,
    METHOD_BAYESIAN: _run_bayesian,
    METHOD_COST_AWARE: _run_cost_aware,
}


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Configuration")

    st.sidebar.subheader("Dataset")
    st.sidebar.write("Breast Cancer")

    st.sidebar.subheader("Model")
    model = st.sidebar.selectbox("Model", ["Random Forest"], index=0)

    st.sidebar.subheader("Optimization Method")
    method = st.sidebar.selectbox("Method", ALL_METHODS, index=0)

    st.sidebar.subheader("Metric")
    metric = st.sidebar.selectbox("Metric", ["F1"], index=0)

    st.sidebar.subheader("Time Budget")
    budget = st.sidebar.slider(
        "Budget (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        step=10,
        help="Applies to search-based methods. Ignored by the default baseline.",
    )

    st.sidebar.subheader("Random Seed")
    seed = st.sidebar.number_input(
        "Seed", min_value=0, value=42, step=1, format="%d"
    )

    run_clicked = st.sidebar.button("Run Optimization", type="primary")

    return {
        "model": model,
        "method": method,
        "metric": metric,
        "budget": float(budget),
        "seed": int(seed),
        "run_clicked": run_clicked,
    }


def _render_results(result: Dict[str, Any], method: str) -> None:
    st.subheader("Best Result")
    c1, c2, c3 = st.columns(3)

    best_f1 = result.get("best_f1")
    c1.metric(
        "Best F1",
        f"{best_f1:.4f}" if best_f1 is not None else "N/A",
    )
    c2.metric("Total Time", f"{result['total_time']:.4f} s")
    c3.metric("Number of Trials", str(result["num_trials"]))

    st.subheader("Best Parameters")
    best_params = result.get("best_parameters") or {}
    if best_params:
        st.code(_format_params(best_params), language="text")
    else:
        st.info("No hyperparameter search was performed by this method.")

    if result.get("is_cost_aware"):
        util = result.get("best_utility")
        if util is not None:
            st.metric("Best Utility (F1 / predicted cost)", f"{util:.4f}")

    # Full metric breakdown when available.
    metrics = result.get("metrics") or {}
    if metrics:
        st.subheader("Full Test Metrics")
        st.dataframe(
            pd.DataFrame([metrics]),
            use_container_width=True,
        )

    trials: List[TrialRecord] = result.get("trials") or []
    if not trials:
        return

    st.subheader("Trial Results")
    df = _trials_to_dataframe(trials, cost_aware=result.get("is_cost_aware", False))
    st.dataframe(df, use_container_width=True)

    conv = _convergence_dataframe(trials)
    if not conv.empty:
        st.subheader("Optimization Convergence")
        st.line_chart(conv.set_index("Trial")["Best F1 so far"])


def main() -> None:
    st.set_page_config(page_title="TimeTuneML", page_icon="⏱", layout="wide")

    st.title("TimeTuneML")
    st.caption("Automatic Hyperparameter Tuning Under Time Constraints")

    cfg = _render_sidebar()

    if not cfg["run_clicked"]:
        st.info(
            "Configure the options in the sidebar and click "
            "**Run Optimization** to start."
        )
        return

    # Load + validate the dataset once per run.
    try:
        X, y, feature_names, _target_names = load_breast_cancer_dataset()
        validate_dataset(X, y)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load dataset: {exc}")
        return

    runner = _RUNNERS[cfg["method"]]

    with st.spinner(f"Running {cfg['method']} ..."):
        try:
            result = runner(
                X,
                y,
                feature_names,
                seed=cfg["seed"],
                budget_seconds=cfg["budget"],
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimization failed: {exc}")
            return

    if result.get("best_f1") is None and not result.get("trials"):
        st.warning("Optimization completed but no successful trial was recorded.")
        return

    _render_results(result, cfg["method"])


if __name__ == "__main__":
    main()