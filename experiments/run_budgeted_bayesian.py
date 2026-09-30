"""Run budgeted Bayesian (Optuna TPE) optimization for TimeTuneML.

Usage
-----
    python experiments/run_budgeted_bayesian.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script: add `src/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timetuneml.data.loader import load_breast_cancer_dataset
from timetuneml.data.validation import validate_dataset
from timetuneml.models.search_spaces import get_random_forest_search_space
from timetuneml.optimization.bayesian import BayesianOptimizationOptimizer

SEED = 42
BUDGET_SECONDS = 30.0


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Budgeted Bayesian Optimization")
    print("=" * 60)
    print(f"\nBudget: {BUDGET_SECONDS:.2f} seconds\n")

    # 1. Load + validate (reuses Milestone 1 components).
    X, y, feature_names, _target_names = load_breast_cancer_dataset()
    validate_dataset(X, y)

    # 2. Build budgeted Bayesian optimizer.
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=SEED,
        budget_seconds=BUDGET_SECONDS,
    )

    # 3. Run until the budget is exhausted.
    optimizer.optimize()

    # 4. Per-trial report.
    for trial in optimizer.trials:
        print(f"Trial {trial.trial_id}")
        if trial.status == "COMPLETE":
            print(f"Parameters           : {_format_params(trial.parameters)}")
            print(f"F1                   : {trial.score:.4f}")
            print(f"Duration             : {trial.duration_seconds:.4f} s")
            print(f"Cumulative Time      : {trial.cumulative_time_seconds:.4f} s")
            print(f"Status               : {trial.status}")
        else:
            print(f"Parameters           : {_format_params(trial.parameters)}")
            print(f"Duration             : {trial.duration_seconds:.4f} s")
            print(f"Cumulative Time      : {trial.cumulative_time_seconds:.4f} s")
            print(f"Status               : {trial.status}")
            print(f"Error                : {trial.error}")
        print()

    # 5. Budget summary.
    elapsed = (
        optimizer.trials[-1].cumulative_time_seconds
        if optimizer.trials
        else 0.0
    )

    print("=" * 60)
    print("Budget Summary")
    print("=" * 60)
    print(f"Configured Budget : {BUDGET_SECONDS:.2f} s")
    print(f"Elapsed Time      : {elapsed:.4f} s")
    print(f"Trials Completed  : {len(optimizer.trials)}")
    print(f"Budget Exhausted  : {'YES' if optimizer.budget_exhausted else 'NO'}")
    print()

    # 6. Best trial.
    best = optimizer.best_trial()
    print("=" * 60)
    print("Best Trial")
    print("=" * 60)
    if best is None:
        print("No successful trials were recorded.")
    else:
        print(f"Trial ID       : {best.trial_id}")
        print(f"Best F1        : {best.score:.4f}")
        print(f"Duration       : {best.duration_seconds:.4f} s")
        print(f"Cumulative Time: {best.cumulative_time_seconds:.4f} s")
        print(f"Parameters     : {best.parameters}")

    print("\nDone.")


if __name__ == "__main__":
    main()