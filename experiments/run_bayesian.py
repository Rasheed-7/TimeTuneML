"""Run Bayesian (Optuna TPE) hyperparameter optimization for TimeTuneML.

Usage
-----
    python experiments/run_bayesian.py
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

N_TRIALS = 10
SEED = 42


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Bayesian Optimization")
    print("=" * 60)

    # 1. Load + validate (reuses Milestone 1 components).
    X, y, feature_names, _target_names = load_breast_cancer_dataset()
    validate_dataset(X, y)

    # 2. Build optimizer reusing the existing search space.
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        n_trials=N_TRIALS,
        random_state=SEED,
    )

    # 3. Run search.
    optimizer.optimize()

    # 4. Print every trial.
    print()
    for trial in optimizer.trials:
        print(f"Trial {trial.trial_id}")
        if trial.status == "COMPLETE":
            print(f"Parameters : {_format_params(trial.parameters)}")
            print(f"F1         : {trial.score:.4f}")
            print(f"Duration   : {trial.duration_seconds:.4f} s")
        else:
            print(f"Parameters : {_format_params(trial.parameters)}")
            print(f"Status     : {trial.status}")
            print(f"Error      : {trial.error}")
        print()

    # 5. Convergence: best F1 observed up to and including each trial.
    print("=" * 60)
    print("Convergence")
    print("=" * 60)
    best_so_far = float("-inf")
    for trial in optimizer.trials:
        if trial.status == "COMPLETE" and trial.score is not None:
            best_so_far = max(best_so_far, trial.score)
            print(f"Trial {trial.trial_id:<3}: {best_so_far:.4f}")
        else:
            if best_so_far == float("-inf"):
                print(f"Trial {trial.trial_id:<3}: (no successful trial yet)")
            else:
                print(f"Trial {trial.trial_id:<3}: {best_so_far:.4f}")
    print()

    # 6. Best trial summary.
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
        print(f"Parameters     : {best.parameters}")

    print()
    print(f"Total search duration: {optimizer.total_duration_seconds:.4f} s")
    print("\nDone.")


if __name__ == "__main__":
    main()