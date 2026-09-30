"""Dataset loading utilities for TimeTuneML.

Supports the following built-in sklearn datasets:

    * breast_cancer  (binary classification, 569 samples, 30 features)
    * iris           (3-class classification, 150 samples, 4 features)
    * wine           (3-class classification, 178 samples, 13 features)

All loaders return the same tuple shape consumed by the existing
preprocessing/evaluation pipeline:

    (X, y, feature_names, target_names)

Nothing dataset-specific is done inside the optimizers — they only ever
see a NumPy feature matrix and a NumPy label vector.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
# Map short name -> (loader callable, human-readable name)
_DATASET_REGISTRY = {
    "breast_cancer": (load_breast_cancer, "Breast Cancer"),
    "iris":          (load_iris,          "Iris"),
    "wine":          (load_wine,          "Wine"),
}

SUPPORTED_DATASETS: List[str] = list(_DATASET_REGISTRY.keys())


# ----------------------------------------------------------------------
# Generic loader
# ----------------------------------------------------------------------
def load_dataset(
    name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load one of the built-in datasets by short name.

    Parameters
    ----------
    name : str
        One of ``SUPPORTED_DATASETS`` (``"breast_cancer"``, ``"iris"``,
        ``"wine"``).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_names : list of str
    target_names : list of str
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    loader, _pretty = _DATASET_REGISTRY[name]
    bunch = loader()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target)
    feature_names = list(bunch.feature_names)
    target_names = list(bunch.target_names)
    return X, y, feature_names, target_names


def dataset_display_name(name: str) -> str:
    """Return the human-readable name for a dataset key."""
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}.")
    return _DATASET_REGISTRY[name][1]


# ----------------------------------------------------------------------
# Backwards-compatible wrapper (Milestone 1 API — do not remove)
# ----------------------------------------------------------------------
def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Load the built-in sklearn breast cancer dataset.

    Kept as a thin wrapper so existing code (Milestones 1–8, Streamlit,
    benchmark scripts) continues to work unchanged.
    """
    return load_dataset("breast_cancer")


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------
def get_dataset_metadata(name: str) -> Dict[str, object]:
    """Return basic metadata for a dataset.

    Returns a dict with keys:
        name         : human-readable name (e.g. "Breast Cancer")
        key          : short name (e.g. "breast_cancer")
        n_samples    : int
        n_features   : int
        n_classes    : int
        class_names  : list of str
    """
    X, y, feature_names, target_names = load_dataset(name)
    return {
        "name": dataset_display_name(name),
        "key": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_names": list(target_names),
    }