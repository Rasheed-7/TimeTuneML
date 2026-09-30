"""Run the TimeTuneML baseline Random Forest experiment.

Usage
-----
    python experiments/run_baseline.py
    python experiments/run_baseline.py --dataset iris
    python experiments/run_baseline.py --dataset wine
    python experiments/run_baseline.py --dataset breast_cancer   # default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script: add `src/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timetuneml.data.loader import (
    SUPPORTED_DATASETS,
    dataset_display_name,
    load_dataset,
)
from timetuneml.data.validation import describe_dataset, validate_dataset
from timetuneml.evaluation.evaluator import run_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TimeTuneML baseline Random Forest experiment."
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="breast_cancer",
        help="Built-in dataset to use (default: breast_cancer).",
    )
    args = parser.parse_args()

    display_name = dataset_display_name(args.dataset)

    print("=" * 60)
    print("TimeTuneML — Baseline")
    print("=" * 60)

    # 1. Load
    X, y, feature_names, target_names = load_dataset(args.dataset)

    # 2. Validate
    validate_dataset(X, y)
    info = describe_dataset(X, y, target_names)

    print(f"\nDataset: {display_name}")
    print("-" * 60)
    print(f"Samples        : {info['n_samples']}")
    print(f"Features       : {info['n_features']}")
    print(f"Classes        : {info['n_classes']} -> {info['class_names']}")
    print("Class counts   :")
    for name, count in info["class_counts"].items():
        print(f"    {name:<12}: {count}")

    # 3. Run baseline
    print("\nRunning baseline Random Forest ...")
    result = run_baseline_experiment(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
    )

    # 4. Report
    print("\nTraining summary")
    print("-" * 60)
    print(f"Train samples  : {result['n_train']}")
    print(f"Test samples   : {result['n_test']}")
    print(f"Random state   : {result['random_state']}")
    print(f"Train duration : {result['train_duration_seconds']:.4f} seconds")

    print("\nTest metrics")
    print("-" * 60)
    for name, value in result["metrics"].items():
        print(f"{name:<10}: {value:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()