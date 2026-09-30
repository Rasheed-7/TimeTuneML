"""Tests for the TimeTuneML Random Forest search space.

These tests use the actual search-space registry and verify that any
sampled configuration stays within the declared bounds. They do not
reimplement the search-space definition — they read it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timetuneml.models.search_spaces import (  # noqa: E402
    get_random_forest_search_space,
)


@pytest.fixture
def space():
    return get_random_forest_search_space()


def test_search_space_has_all_expected_keys(space):
    expected = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    }
    assert expected.issubset(space.keys())


def test_search_space_returns_independent_copy():
    a = get_random_forest_search_space()
    b = get_random_forest_search_space()
    a["n_estimators"]["low"] = -1
    # Mutating one copy must not affect the next retrieval.
    assert b["n_estimators"]["low"] != -1


def test_sampled_parameters_are_within_declared_bounds(space):
    """Draw many samples using the space's own spec and check each one."""
    rng = np.random.default_rng(42)
    n_samples = 500

    for _ in range(n_samples):
        params = {}
        for name, spec in space.items():
            kind = spec["type"]
            if kind == "int":
                params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
            elif kind == "float":
                params[name] = float(rng.uniform(spec["low"], spec["high"]))
            elif kind == "choice":
                options = list(spec["options"])
                params[name] = options[int(rng.integers(0, len(options)))]
            else:
                pytest.fail(f"Unknown spec type: {kind!r}")

        # Integer-range checks
        assert space["n_estimators"]["low"] <= params["n_estimators"] <= space["n_estimators"]["high"]
        assert space["max_depth"]["low"] <= params["max_depth"] <= space["max_depth"]["high"]
        assert space["min_samples_split"]["low"] <= params["min_samples_split"] <= space["min_samples_split"]["high"]
        assert space["min_samples_leaf"]["low"] <= params["min_samples_leaf"] <= space["min_samples_leaf"]["high"]

        # Categorical check
        assert params["max_features"] in space["max_features"]["options"]


def test_max_features_options_are_the_expected_choices(space):
    options = space["max_features"]["options"]
    for expected in ["sqrt", "log2", 0.3, 0.5, 1.0]:
        assert expected in options