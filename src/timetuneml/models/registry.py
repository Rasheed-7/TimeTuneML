"""Model registry for TimeTuneML.

Currently only exposes the baseline Random Forest classifier used in
Milestone 1. Later milestones will add more models and search strategies.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def get_baseline_random_forest(
    random_state: int = RANDOM_STATE,
    n_estimators: int = 200,
) -> RandomForestClassifier:
    # n_jobs=1: the baseline is a single fit, so parallel workers add
    # overhead without meaningful speedup on these small datasets.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )