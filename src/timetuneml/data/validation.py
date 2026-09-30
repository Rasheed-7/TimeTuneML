"""Dataset validation utilities for TimeTuneML."""

from __future__ import annotations

import numpy as np


def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Validate a feature matrix and target vector.

    Performs the following checks:
      * target exists (not None)
      * X and y have matching row counts
      * no missing values in X
      * no missing values in y

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if y is None:
        raise ValueError("Target vector y is missing (None).")

    if X is None:
        raise ValueError("Feature matrix X is missing (None).")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row count mismatch: X has {X.shape[0]} rows, "
            f"y has {y.shape[0]} rows."
        )

    if X.size == 0 or y.size == 0:
        raise ValueError("Dataset is empty (zero samples).")

    if np.isnan(X).any():
        n_missing = int(np.isnan(X).sum())
        raise ValueError(f"Feature matrix X contains {n_missing} missing value(s).")

    if np.isnan(y).any():
        n_missing = int(np.isnan(y).sum())
        raise ValueError(f"Target vector y contains {n_missing} missing value(s).")


def describe_dataset(X: np.ndarray, y: np.ndarray, target_names: list) -> dict:
    """Return a small summary dictionary describing the dataset."""
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_names": list(target_names),
        "class_counts": {
            name: int((y == idx).sum())
            for idx, name in enumerate(target_names)
        },
    }