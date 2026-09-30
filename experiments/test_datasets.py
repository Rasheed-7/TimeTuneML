"""Sanity-check all built-in TimeTuneML datasets.

For each dataset, verify that:
    * loading succeeds
    * validation succeeds
    * train/test split succeeds
    * preprocessing pipeline fits on train and transforms test
    * Random Forest trains
    * F1 is computed on the test set

Usage
-----
    python experiments/test_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Allow running as a plain script: add `src/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timetuneml.data.loader import (
    SUPPORTED_DATASETS,
    get_dataset_metadata,
    load_dataset,
)
from timetuneml.data.validation import describe_dataset, validate_dataset
from timetuneml.evaluation.metrics import compute_classification_metrics
from timetuneml.models.registry import get_baseline_random_forest
from timetuneml.preprocessing.pipeline import build_preprocessing_pipeline


def _check_dataset(name: str) -> dict:
    """Run the full train/preprocess/fit/evaluate sanity path.

    Returns a dict summarising the outcome for reporting.
    """
    outcome = {
        "key": name,
        "status": "OK",
        "error": None,
        "f1": None,
    }
    try:
        X, y, feature_names, target_names = load_dataset(name)

        # 1. Validation
        validate_dataset(X, y)
        describe_dataset(X, y, target_names)  # exercises class-count path

        # 2. Split (same configuration used across the project)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Preprocessing — fit on train only, transform test
        preprocessor = build_preprocessing_pipeline(
            numeric_features=feature_names, categorical_features=None,
        )
        X_train_t = preprocessor.fit_transform(X_train, y_train)
        X_test_t = preprocessor.transform(X_test)

        # 4. Model
        model = get_baseline_random_forest(random_state=42)
        model.fit(X_train_t, y_train)

        # 5. Metrics
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        outcome["f1"] = float(metrics["f1"])
    except Exception as exc:  # noqa: BLE001
        outcome["status"] = "FAILED"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    return outcome


def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Dataset Validation")
    print("=" * 60)
    print()

    # ---- metadata table ------------------------------------------------
    header = (
        "Dataset".ljust(20)
        + "Samples".rjust(10)
        + "Features".rjust(10)
        + "Classes".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for key in SUPPORTED_DATASETS:
        meta = get_dataset_metadata(key)
        print(
            meta["name"].ljust(20)
            + str(meta["n_samples"]).rjust(10)
            + str(meta["n_features"]).rjust(10)
            + str(meta["n_classes"]).rjust(10)
        )
    print()

    # ---- full sanity path per dataset ---------------------------------
    print("Sanity checks (split / preprocess / fit / F1)")
    print("-" * 60)

    all_ok = True
    for key in SUPPORTED_DATASETS:
        result = _check_dataset(key)
        meta = get_dataset_metadata(key)
        if result["status"] == "OK":
            print(
                f"{meta['name']:<20} OK    F1={result['f1']:.4f}"
            )
        else:
            all_ok = False
            print(
                f"{meta['name']:<20} FAILED  {result['error']}"
            )

    print()
    if all_ok:
        print("All datasets passed the sanity check.")
    else:
        print("One or more datasets failed. See errors above.")


if __name__ == "__main__":
    main()