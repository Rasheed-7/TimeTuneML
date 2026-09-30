"""Baseline evaluation pipeline for TimeTuneML.

The evaluator is responsible for the *entire* train/test workflow:

    load -> validate -> split -> fit preprocess (train only)
         -> fit model (train only) -> predict (test only) -> metrics

Preprocessing is fitted **only** on the training split and applied to
the test split via ``transform``. This guarantees no leakage.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from timetuneml.evaluation.metrics import compute_classification_metrics
from timetuneml.models.registry import get_baseline_random_forest, RANDOM_STATE
from timetuneml.preprocessing.pipeline import build_preprocessing_pipeline


def run_baseline_experiment(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features: Sequence[str],
    categorical_features: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    """Run the baseline Random Forest experiment.

    Steps
    -----
    1. Split into train/test (stratified).
    2. Build the preprocessing pipeline (unfitted).
    3. Fit the preprocessing on the training features only.
    4. Fit the Random Forest on the transformed training features.
    5. Transform the test features with the *already fitted* preprocessor.
    6. Evaluate on the test set and record training duration.

    Returns
    -------
    dict
        Contains ``metrics`` (dict), ``train_duration_seconds`` (float),
        ``n_train``, ``n_test``, and ``random_state``.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = get_baseline_random_forest(random_state=random_state)

    # ---- fit only on training data -------------------------------------
    start = time.perf_counter()

    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    model.fit(X_train_transformed, y_train)

    train_duration = time.perf_counter() - start
    # --------------------------------------------------------------------

    # ---- transform test data using the fitted preprocessor -------------
    X_test_transformed = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    return {
        "metrics": metrics,
        "train_duration_seconds": float(train_duration),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "random_state": int(random_state),
    }