"""Search-space registry for TimeTuneML.

Each search space is a plain dict mapping hyperparameter name to a
sampler specification. The Random Search optimizer reads these specs to
draw values. Keeping this as data (not code) makes it easy to add new
model families later.
"""

from __future__ import annotations

from typing import Any, Dict

# A search-space entry is a dict with a "type" key and type-specific keys:
#   {"type": "int",   "low": int, "high": int}                 # inclusive
#   {"type": "float", "low": float, "high": float}             # inclusive
#   {"type": "choice", "options": [...]}                       # any values

RANDOM_FOREST_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "n_estimators":      {"type": "int",    "low": 50,  "high": 500},
    "max_depth":         {"type": "int",    "low": 3,   "high": 30},
    "min_samples_split": {"type": "int",    "low": 2,   "high": 20},
    "min_samples_leaf":  {"type": "int",    "low": 1,   "high": 10},
    "max_features":      {"type": "choice", "options": ["sqrt", "log2", 0.3, 0.5, 1.0]},
}


def get_random_forest_search_space() -> Dict[str, Dict[str, Any]]:
    """Return a fresh copy of the Random Forest search space.

    A copy is returned so callers can safely modify their local version
    without affecting the registry.
    """
    return {
        name: dict(spec)
        for name, spec in RANDOM_FOREST_SEARCH_SPACE.items()
    }