"""Lightweight tests for the TimeTuneML data loading and validation layer.

These tests exercise the *actual* loader and validation functions used by
the project. They do not reimplement any data logic. They do not train
models, run optimizers, or touch the results/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timetuneml.data.loader import (  # noqa: E402
    SUPPORTED_DATASETS,
    get_dataset_metadata,
    load_dataset,
)
from timetuneml.data.validation import validate_dataset  # noqa: E402


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_supported_dataset_loads(name):
    X, y, feature_names, target_names = load_dataset(name)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] > 0
    assert X.shape[1] == len(feature_names) > 0
    assert len(target_names) >= 2


@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_loaded_arrays_are_finite_and_numeric(name):
    X, y, _, _ = load_dataset(name)
    assert np.issubdtype(X.dtype, np.number)
    assert np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.number)
    assert np.all(np.isfinite(X))


def test_unknown_dataset_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_real_dataset")


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_metadata_matches_loaded_arrays(name):
    meta = get_dataset_metadata(name)
    X, y, _, _ = load_dataset(name)
    assert meta["n_samples"] == X.shape[0]
    assert meta["n_features"] == X.shape[1]
    assert meta["n_classes"] == len(np.unique(y))


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_validation_accepts_supported_datasets(name):
    X, y, _, _ = load_dataset(name)
    validate_dataset(X, y)  # should not raise


def test_validation_rejects_row_count_mismatch():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="mismatch"):
        validate_dataset(X, y)


def test_validation_rejects_nan_values():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(X, y)


def test_validation_rejects_empty_dataset():
    X = np.empty((0, 2))
    y = np.empty((0,))
    with pytest.raises(ValueError):
        validate_dataset(X, y)


# ----------------------------------------------------------------------
# Train/test split — verify no leakage through the split interface
# ----------------------------------------------------------------------
def test_train_test_split_produces_disjoint_index_sets():
    """A stratified split must not reuse rows across train and test."""
    from sklearn.model_selection import train_test_split

    X, y, _, _ = load_dataset("breast_cancer")
    # Add an index column so we can identify rows.
    idx = np.arange(X.shape[0]).reshape(-1, 1)
    X_with_idx = np.hstack([X, idx])

    X_tr, X_te, _, _ = train_test_split(
        X_with_idx, y, test_size=0.2, random_state=42, stratify=y,
    )
    train_ids = set(X_tr[:, -1].astype(int).tolist())
    test_ids = set(X_te[:, -1].astype(int).tolist())
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == X.shape[0]