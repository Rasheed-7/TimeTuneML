"""Run Time-Aware (cost-aware) Bayesian optimization for TimeTuneML.

Usage
-----
    python experiments/run_cost_aware.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
WARMUP_TRIALS = 5


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Time-Aware Bayesian Optimization")
    print("=" * 60)
    print(f"\nBudget: {BUDGET_SECONDS:.2f} seconds")
    print(f"Warmup trials: {WARMUP_TRIALS}\n")

    # 1. Load + validate (reuses Milestone 1 components).
    X, y, feature_names, _target_names = load_breast_cancer_dataset()
    validate_dataset(X, y)

    # 2. Build cost-aware optimizer.
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=SEED,
        budget_seconds=BUDGET_SECONDS,
        cost_aware=True,
        warmup_trials=WARMUP_TRIALS,
    )

    # 3. Run.
    optimizer.optimize()

    # 4. Per-trial report.
    for trial in optimizer.trials:
        print(f"Trial {trial.trial_id}")
        print(f"Parameters             : {_format_params(trial.parameters)}")
        if trial.status == "COMPLETE":
            print(f"F1                     : {trial.score:.4f}")
            print(f"Actual Duration        : {trial.duration_seconds:.4f} s")
            if trial.estimated_duration_seconds is not None:
                print(f"Estimated Duration     : {trial.estimated_duration_seconds:.4f} s")
            else:
                print("Estimated Duration     : N/A")
            if trial.utility is not None:
                print(f"Utility                : {trial.utility:.4f}")
            else:
                print("Utility                : N/A")
            print(f"Cumulative Time        : {trial.cumulative_time_seconds:.4f} s")
            print(f"Status                 : {trial.status}")
        else:
            print(f"Actual Duration        : {trial.duration_seconds:.4f} s")
            print("Estimated Duration     : N/A")
            print("Utility                : N/A")
            print(f"Cumulative Time        : {trial.cumulative_time_seconds:.4f} s")
            print(f"Status                 : {trial.status}")
            print(f"Error                  : {trial.error}")
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

    # 6. Best trial (by F1).
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
    print()

    # 7. Cost-aware summary.
    completed = [t for t in optimizer.trials if t.status == "COMPLETE"]
    warmup_done = min(WARMUP_TRIALS, len(completed))
    cost_aware_done = max(0, len(completed) - warmup_done)
    avg_time = (
        sum(t.duration_seconds for t in completed) / len(completed)
        if completed
        else 0.0
    )
    best_util_trial = optimizer.best_utility_trial()
    best_util = best_util_trial.utility if best_util_trial is not None else None
    best_f1 = best.score if best is not None else None

    print("=" * 60)
    print("Cost-Aware Summary")
    print("=" * 60)
    print(f"Warmup Trials       : {warmup_done}")
    print(f"Cost-Aware Trials   : {cost_aware_done}")
    print(f"Average Trial Time  : {avg_time:.4f} s")
    print(
        "Best Utility        : "
        + (f"{best_util:.4f}" if best_util is not None else "N/A")
    )
    print(
        "Best F1             : "
        + (f"{best_f1:.4f}" if best_f1 is not None else "N/A")
    )
    print("\nDone.")


if __name__ == "__main__":
    main()