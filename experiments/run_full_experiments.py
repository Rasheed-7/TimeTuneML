"""Run the TimeTuneML experiment matrix.

Modes
-----
    --test     : seed=42, budget=30, breast_cancer only (4 runs)
    --pilot    : seed=42, budget=30, all three datasets (12 runs)
    --final    : 3 datasets x 4 methods x 3 seeds x 3 budgets (108 runs)
    (none)     : legacy full matrix (kept for backward compatibility)

The script is resumable: before every run it checks whether the same
(dataset, method, seed, budget) combination already exists in
``results/tables/full_experiments.csv`` with status "completed", and
skips it if so. Results are flushed to disk after every experiment, so
an interrupted run can be restarted safely.

Usage
-----
    python experiments/run_full_experiments.py --test
    python experiments/run_full_experiments.py --pilot
    python experiments/run_full_experiments.py --final
    python experiments/run_full_experiments.py --final --dry-run

Design notes
------------
* The Default RF is a single baseline training run; its "budget" is
  recorded for bookkeeping only and is not consumed.
* Search-based methods respect their configured time budget via the
  existing ``budget_seconds`` parameter.
* No optimizer code is modified. No test set is used during optimisation.
* No winner is declared; the summary is descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

# sklearn/joblib replay warnings from worker threads via warn_explicit,
# which bypasses the message-regex filter. Blanket-suppress UserWarning
# from the sklearn and joblib modules at the warnings-category level.
warnings.filterwarnings("ignore", category=UserWarning, module=r".*sklearn.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r".*joblib.*")
import numpy as np

# ---- make src/ importable ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---- silence the sklearn/joblib parallel warning ----------------------
# It is emitted through the logging system (not warnings.warn), so we
# suppress it on both surfaces. Harmless but noisy at thousands of fits.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
for _logger_name in ("joblib", "sklearn", "sklearn.utils.parallel"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from timetuneml.data.loader import (  # noqa: E402
    SUPPORTED_DATASETS,
    dataset_display_name,
    load_dataset,
)
from timetuneml.data.validation import validate_dataset  # noqa: E402
from timetuneml.evaluation.evaluator import run_baseline_experiment  # noqa: E402
from timetuneml.models.search_spaces import get_random_forest_search_space  # noqa: E402
from timetuneml.optimization.bayesian import BayesianOptimizationOptimizer  # noqa: E402
from timetuneml.optimization.random_search import RandomSearchOptimizer  # noqa: E402


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
METHOD_DEFAULT = "Default RF"
METHOD_RANDOM = "Random Search"
METHOD_BAYESIAN = "Bayesian Optimization"
METHOD_COST_AWARE = "Time-Aware Bayesian Optimization"

METHODS = [METHOD_DEFAULT, METHOD_RANDOM, METHOD_BAYESIAN, METHOD_COST_AWARE]

# --- Legacy full matrix (kept for backward compatibility) -------------
BUDGETS_FULL = [30, 60, 120, 300, 600]
SEEDS_FULL = [42, 43, 44, 45, 46]

# --- Pilot -------------------------------------------------------------
BUDGETS_PILOT = [30]
SEEDS_PILOT = [42]

# --- Sanity ------------------------------------------------------------
BUDGETS_TEST = [30]
SEEDS_TEST = [42]

# --- Final milestone matrix --------------------------------------------
# 3 datasets x 4 methods x 3 seeds x 3 budgets = 108 experiments.
BUDGETS_FINAL = [30, 120, 600]
SEEDS_FINAL = [42, 43, 44]

# --- Dataset selections ------------------------------------------------
DATASETS_FULL = list(SUPPORTED_DATASETS)   # breast_cancer, iris, wine
DATASETS_PILOT = list(SUPPORTED_DATASETS)
DATASETS_TEST = ["breast_cancer"]
DATASETS_FINAL = list(SUPPORTED_DATASETS)

WARMUP_TRIALS = 5
TARGET_F1 = 0.9655

MODEL_NAME = "Random Forest"
METRIC_NAME = "F1"

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FULL_CSV = RESULTS_DIR / "full_experiments.csv"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"

FULL_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "budget_seconds",
    "best_f1",
    "total_time_seconds",
    "num_trials",
    "best_parameters",
    "best_utility",
    "average_trial_time",
    "time_to_target",
    "model",
    "metric",
    "status",
    "error",
]

SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "budget_seconds",
    "n_runs",
    "mean_best_f1",
    "std_best_f1",
    "median_best_f1",
    "mean_total_time",
    "std_total_time",
    "mean_num_trials",
    "mean_average_trial_time",
]


# ----------------------------------------------------------------------
# Per-method runners (dataset-agnostic)
# ----------------------------------------------------------------------
def _run_default(X, y, feature_names, seed: int, budget_seconds: float) -> Dict[str, Any]:
    result = run_baseline_experiment(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        random_state=seed,
    )
    return {
        "best_f1": float(result["metrics"]["f1"]),
        "total_time_seconds": float(result["train_duration_seconds"]),
        "num_trials": 1,
        "best_parameters": {},
        "best_utility": None,
        "average_trial_time": None,
        "time_to_target": None,
    }


def _run_random_search(X, y, feature_names, seed: int, budget_seconds: float) -> Dict[str, Any]:
    optimizer = RandomSearchOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    completed = [t for t in optimizer.trials if t.status == "COMPLETE"]
    avg_time = (
        sum(t.duration_seconds for t in completed) / len(completed)
        if completed else None
    )
    return {
        "best_f1": float(best.score) if best is not None else None,
        "total_time_seconds": float(optimizer.total_duration_seconds),
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "best_utility": None,
        "average_trial_time": float(avg_time) if avg_time is not None else None,
        "time_to_target": _time_to_target(optimizer.trials, TARGET_F1),
    }


def _run_bayesian(X, y, feature_names, seed: int, budget_seconds: float) -> Dict[str, Any]:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
        cost_aware=False,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    completed = [t for t in optimizer.trials if t.status == "COMPLETE"]
    avg_time = (
        sum(t.duration_seconds for t in completed) / len(completed)
        if completed else None
    )
    return {
        "best_f1": float(best.score) if best is not None else None,
        "total_time_seconds": float(optimizer.total_duration_seconds),
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "best_utility": None,
        "average_trial_time": float(avg_time) if avg_time is not None else None,
        "time_to_target": _time_to_target(optimizer.trials, TARGET_F1),
    }


def _run_cost_aware(X, y, feature_names, seed: int, budget_seconds: float) -> Dict[str, Any]:
    optimizer = BayesianOptimizationOptimizer(
        X=X,
        y=y,
        numeric_features=feature_names,
        categorical_features=None,
        search_space=get_random_forest_search_space(),
        random_state=seed,
        budget_seconds=budget_seconds,
        cost_aware=True,
        warmup_trials=WARMUP_TRIALS,
    )
    optimizer.optimize()
    best = optimizer.best_trial()
    best_util_trial = optimizer.best_utility_trial()
    completed = [t for t in optimizer.trials if t.status == "COMPLETE"]
    avg_time = (
        sum(t.duration_seconds for t in completed) / len(completed)
        if completed else None
    )
    return {
        "best_f1": float(best.score) if best is not None else None,
        "total_time_seconds": float(optimizer.total_duration_seconds),
        "num_trials": len(optimizer.trials),
        "best_parameters": best.parameters if best is not None else {},
        "best_utility": (
            float(best_util_trial.utility)
            if best_util_trial is not None and best_util_trial.utility is not None
            else None
        ),
        "average_trial_time": float(avg_time) if avg_time is not None else None,
        "time_to_target": _time_to_target(optimizer.trials, TARGET_F1),
    }


_RUNNERS = {
    METHOD_DEFAULT: _run_default,
    METHOD_RANDOM: _run_random_search,
    METHOD_BAYESIAN: _run_bayesian,
    METHOD_COST_AWARE: _run_cost_aware,
}


# ----------------------------------------------------------------------
# Time-to-target helper
# ----------------------------------------------------------------------
def _time_to_target(trials, target_f1: float) -> Optional[float]:
    best_so_far = float("-inf")
    for t in trials:
        if t.status != "COMPLETE" or t.score is None:
            continue
        if t.score > best_so_far:
            best_so_far = t.score
        if best_so_far >= target_f1:
            return float(t.cumulative_time_seconds)
    return None


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def _load_existing_results(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _normalise_dataset(value: Optional[str]) -> str:
    """Backward-compat: historical rows without a dataset default to breast_cancer."""
    if value is None or value == "":
        return "breast_cancer"
    return value


def _already_completed(
    rows: List[Dict[str, Any]],
    dataset: str,
    method: str,
    seed: int,
    budget: int,
) -> bool:
    target = (dataset, method, str(seed), str(budget))
    for r in rows:
        key = (
            _normalise_dataset(r.get("dataset")),
            r.get("method", ""),
            str(r.get("seed", "")),
            str(r.get("budget_seconds", "")),
        )
        if key == target and r.get("status", "").lower() == "completed":
            return True
    return False


def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single row and flush immediately (crash-safe)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    write_header = not file_exists
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FULL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FULL_COLUMNS})
        f.flush()


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
def _write_summary(csv_path: Path, summary_path: Path) -> None:
    rows = _load_existing_results(csv_path)
    completed = [r for r in rows if r.get("status", "").lower() == "completed"]

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in completed:
        try:
            key = (
                _normalise_dataset(r.get("dataset")),
                r["method"],
                float(r["budget_seconds"]),
            )
        except (KeyError, ValueError, TypeError):
            continue
        groups.setdefault(key, []).append(r)

    def _f(row, key) -> Optional[float]:
        v = row.get(key, "")
        if v in ("", None):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out_rows: List[Dict[str, Any]] = []
    for (dataset, method, budget), group in sorted(groups.items()):
        f1s = [v for v in (_f(r, "best_f1") for r in group) if v is not None]
        times = [v for v in (_f(r, "total_time_seconds") for r in group) if v is not None]
        trials = [v for v in (_f(r, "num_trials") for r in group) if v is not None]
        avg_trial_times = [
            v for v in (_f(r, "average_trial_time") for r in group) if v is not None
        ]

        def _mean(xs):
            return float(np.mean(xs)) if xs else None

        def _std(xs):
            if not xs or len(xs) < 2:
                return None
            return float(np.std(xs, ddof=0))

        def _median(xs):
            return float(np.median(xs)) if xs else None

        out_rows.append({
            "dataset": dataset,
            "method": method,
            "budget_seconds": budget,
            "n_runs": len(group),
            "mean_best_f1": _mean(f1s),
            "std_best_f1": _std(f1s),
            "median_best_f1": _median(f1s),
            "mean_total_time": _mean(times),
            "std_total_time": _std(times),
            "mean_num_trials": _mean(trials),
            "mean_average_trial_time": _mean(avg_trial_times),
        })

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Summary written to: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary rows       : {len(out_rows)}")


# ----------------------------------------------------------------------
# Matrix construction
# ----------------------------------------------------------------------
def _build_matrix(mode: str):
    """Return a list of (dataset, method, seed, budget) tuples."""
    if mode == "test":
        datasets, seeds, budgets = DATASETS_TEST, SEEDS_TEST, BUDGETS_TEST
    elif mode == "pilot":
        datasets, seeds, budgets = DATASETS_PILOT, SEEDS_PILOT, BUDGETS_PILOT
    elif mode == "final":
        datasets, seeds, budgets = DATASETS_FINAL, SEEDS_FINAL, BUDGETS_FINAL
    else:
        datasets, seeds, budgets = DATASETS_FULL, SEEDS_FULL, BUDGETS_FULL

    matrix = []
    for dataset in datasets:
        for seed in seeds:
            for budget in budgets:
                for method in METHODS:
                    matrix.append((dataset, method, seed, budget))
    return matrix


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the TimeTuneML experiment matrix."
    )

    # Mode flags are mutually exclusive.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test",
        action="store_true",
        help="Sanity mode: seed=42, budget=30, breast_cancer only (4 runs).",
    )
    group.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: seed=42, budget=30, all three datasets (12 runs).",
    )
    group.add_argument(
        "--final",
        action="store_true",
        help="Final milestone matrix: 3 datasets x 4 methods x 3 seeds x "
             "3 budgets = 108 runs.",
    )

    # --dry-run is a modifier, not a mode.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment matrix and exit (no experiments run). "
             "Combine with --test, --pilot, or --final.",
    )
    args = parser.parse_args()

    if args.test:
        mode = "test"
        mode_label = "TEST (seed=42, budget=30, breast_cancer only)"
    elif args.pilot:
        mode = "pilot"
        mode_label = "PILOT (seed=42, budget=30, all datasets)"
    elif args.final:
        mode = "final"
        mode_label = "FINAL MATRIX (3 datasets x 4 methods x 3 seeds x 3 budgets)"
    else:
        mode = "full"
        mode_label = "LEGACY FULL MATRIX"

    print("=" * 60)
    print("TimeTuneML — Full Experiments")
    print("=" * 60)
    print(f"Mode       : {mode_label}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Metric     : {METRIC_NAME}")
    print(f"Results    : {FULL_CSV.relative_to(PROJECT_ROOT)}")
    print()

    matrix = _build_matrix(mode)
    total = len(matrix)

    if args.dry_run:
        print(f"Dry run — mode: {mode}")
        print(f"Matrix size: {total}")
        print()
        for i, (dataset, method, seed, budget) in enumerate(matrix, start=1):
            print(
                f"{i:4d}. {dataset:<14} | {method:<34} | seed={seed} | budget={budget}s"
            )
        unique = set(matrix)
        print()
        print(f"Unique combinations: {len(unique)}")
        print(f"Duplicates         : {len(matrix) - len(unique)}")
        return 0

    # Cache dataset arrays so we don't reload them per experiment.
    dataset_cache: Dict[str, tuple] = {}

    def _get_dataset(key: str):
        if key not in dataset_cache:
            X, y, feature_names, _target_names = load_dataset(key)
            validate_dataset(X, y)
            dataset_cache[key] = (X, y, feature_names)
        return dataset_cache[key]

    existing = _load_existing_results(FULL_CSV)

    completed = 0
    failed = 0
    skipped = 0

    for idx, (dataset, method, seed, budget) in enumerate(matrix, start=1):
        display = dataset_display_name(dataset)
        print("=" * 60)
        print(f"Experiment {idx}/{total}")
        print(f"Dataset: {display}")
        print(f"Method: {method}")
        print(f"Seed: {seed}")
        print(f"Budget: {budget} seconds")
        print("=" * 60)
        print(
            f"[START] {dataset} | {method} | seed={seed} | budget={budget}s",
            flush=True,
        )
        experiment_start = time.perf_counter()

        if _already_completed(existing, dataset, method, seed, budget):
            print("Status       : SKIPPED (already completed)")
            print()
            skipped += 1
            continue

        try:
            X, y, feature_names = _get_dataset(dataset)
            result = _RUNNERS[method](
                X, y, feature_names, seed=seed, budget_seconds=float(budget)
            )
            row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "budget_seconds": budget,
                "best_f1": result["best_f1"],
                "total_time_seconds": result["total_time_seconds"],
                "num_trials": result["num_trials"],
                "best_parameters": _params_to_str(result["best_parameters"]),
                "best_utility": result["best_utility"],
                "average_trial_time": result["average_trial_time"],
                "time_to_target": result["time_to_target"],
                "model": MODEL_NAME,
                "metric": METRIC_NAME,
                "status": "completed",
                "error": "",
            }
            _append_row(FULL_CSV, row)
            existing.append({k: ("" if v is None else v) for k, v in row.items()})

            print("Status       : COMPLETE")
            print(f"Best F1      : {_fmt(result['best_f1'])}")
            print(f"Total Time   : {_fmt(result['total_time_seconds'])} s")
            print(f"Trials       : {result['num_trials']}")
            if result.get("time_to_target") is not None:
                print(
                    f"Time-to-Target (F1 >= {TARGET_F1}): "
                    f"{result['time_to_target']:.4f} s"
                )
            elapsed = time.perf_counter() - experiment_start
            print(
                f"[END] {dataset} | {method} | seed={seed} | budget={budget}s "
                f"| elapsed={elapsed:.2f}s",
                flush=True,
            )
            print()
            completed += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user. Preserving completed results.")
            print("Re-run the script to resume.")
            raise

        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "budget_seconds": budget,
                "best_f1": "",
                "total_time_seconds": "",
                "num_trials": "",
                "best_parameters": "",
                "best_utility": "",
                "average_trial_time": "",
                "time_to_target": "",
                "model": MODEL_NAME,
                "metric": METRIC_NAME,
                "status": "failed",
                "error": err,
            }
            _append_row(FULL_CSV, row)
            existing.append({k: ("" if v is None else v) for k, v in row.items()})

            print("Status       : FAILED")
            print(f"Error        : {err}")
            print()
            print(traceback.format_exc())
            elapsed = time.perf_counter() - experiment_start
            print(
                f"[END] {dataset} | {method} | seed={seed} | budget={budget}s "
                f"| elapsed={elapsed:.2f}s | status=FAILED",
                flush=True,
            )
            print()
            failed += 1

    # ---- final report (OUTSIDE the loop) -----------------------------
    banner_title = {
        "test": "TEST COMPLETE",
        "pilot": "PILOT COMPLETE",
        "final": "FINAL EXPERIMENTS COMPLETE",
        "full": "FULL EXPERIMENTS COMPLETE",
    }[mode]
    print("=" * 60)
    print(banner_title)
    print("=" * 60)
    print(f"Total experiments : {total}")
    print(f"Completed         : {completed}")
    print(f"Failed            : {failed}")
    print(f"Skipped           : {skipped}")
    print()
    print(f"Results saved to  : {FULL_CSV.relative_to(PROJECT_ROOT)}")
    print()

    _write_summary(FULL_CSV, SUMMARY_CSV)

    return 0


# ----------------------------------------------------------------------
# Small formatting helpers
# ----------------------------------------------------------------------
def _params_to_str(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    return ";".join(f"{k}={v}" for k, v in params.items())


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    sys.exit(main())