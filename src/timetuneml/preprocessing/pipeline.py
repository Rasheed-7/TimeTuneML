"""Leakage-safe preprocessing pipeline for TimeTuneML.

The pipeline is intentionally designed to support both numeric and
categorical columns even though the initial sklearn dataset is fully
numeric. Preprocessing is meant to be fitted only on training data.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Optional[Sequence[str]] = None,
) -> ColumnTransformer:
    """Build a leakage-safe ColumnTransformer for numeric + categorical data.

    Parameters
    ----------
    numeric_features : sequence of str
        Names of the numeric columns, in the order they appear in X.
    categorical_features : sequence of str, optional
        Names of the categorical columns, in the order they appear in X
        *after* the numeric columns. Defaults to an empty list.

    Returns
    -------
    ColumnTransformer
        A preprocessing transformer. It must be fitted **only** on the
        training set; calling ``.fit`` (or ``.fit_transform``) on the
        test set would cause data leakage.

    Notes
    -----
    The returned ``ColumnTransformer`` uses **positional indices**, not
    string names, for column selection. This is required because
    TimeTuneML feeds raw NumPy arrays (which have no ``.columns``
    attribute) to the pipeline, and ``ColumnTransformer`` can only
    resolve string column names when the input is a pandas DataFrame.
    Numeric columns are assumed to occupy the first positions of the
    input array, followed by the categorical columns.
    """
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features or [])

    # Convert the feature-name lists into positional indices. This is the
    # only column-spec form that reliably works against NumPy arrays.
    numeric_indices = list(range(len(numeric_features)))
    cat_start = len(numeric_features)
    categorical_indices = list(
        range(cat_start, cat_start + len(categorical_features))
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_indices:
        transformers.append(("num", numeric_pipeline, numeric_indices))
    if categorical_indices:
        transformers.append(("cat", categorical_pipeline, categorical_indices))

    if not transformers:
        raise ValueError(
            "At least one of numeric_features or categorical_features "
            "must be provided."
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")