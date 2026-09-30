"""Classification metrics for TimeTuneML.

Handles both binary and multi-class targets transparently:

    * 2 classes  -> binary averaging, ROC-AUC from the positive-class column
    * >2 classes -> macro averaging, ROC-AUC via one-vs-rest on the full
                    probability matrix

The metric name in the returned dict is always ``f1`` (used as the
optimization objective throughout the project); the averaging strategy
is chosen internally, so callers do not need to know how many classes
the dataset has.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Binary vs. multi-class is detected automatically from ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities. Shape (n,) or (n, 2) for binary;
        (n, n_classes) for multi-class.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, roc_auc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError(
            f"compute_classification_metrics requires at least 2 classes; "
            f"got {n_classes}."
        )

    # ------------------------------------------------------------------
    # Averaging strategy
    # ------------------------------------------------------------------
    if n_classes == 2:
        # Binary: keep the original behaviour exactly as in Milestone 1.
        average = "binary"
        if y_proba.ndim == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
    else:
        # Multi-class: macro-averaged precision/recall/F1 and OVR ROC-AUC.
        average = "macro"
        if y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
            raise ValueError(
                f"y_proba must have shape (n_samples, {n_classes}) "
                f"for multi-class targets; got {y_proba.shape}."
            )
        y_score = y_proba  # roc_auc_score handles the multi-class OVR case

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "roc_auc": float(_safe_roc_auc(y_true, y_score, n_classes)),
    }


def _safe_roc_auc(y_true, y_score, n_classes: int) -> float:
    """Compute ROC-AUC, returning NaN if sklearn cannot (e.g. single class).

    Never raises — downstream code should not crash on a degenerate split.
    """
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_score))
        return float(
            roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")