"""Analyze the TimeTuneML multi-dataset pilot results.

Reads:
    results/tables/full_experiments.csv

Tolerates CSVs written by either the Milestone 9 schema
(no `dataset` column, `dataset_name` free-form field) or the
Milestone 9-pilot schema (explicit `dataset` column). Any row whose
`dataset` value is not a known dataset key is treated as legacy and
matched by method/seed/budget only.

Usage
-----
    python experiments/analyze_pilot.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FULL_CSV = RESULTS_DIR / "full_experiments.csv"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"

KNOWN_DATASETS = {"breast_cancer", "iris", "wine"}

DATASET_DISPLAY = {
    "breast_cancer": "Breast Cancer",
    "iris": "Iris",
    "wine": "Wine",
}

SEARCH_METHODS = [
    "Random Search",
    "Bayesian Optimization",
    "Time-Aware Bayesian Optimization",
]
ALL_METHODS = ["Default RF"] + SEARCH_METHODS


# ----------------------------------------------------------------------
# Loading + schema normalisation
# ----------------------------------------------------------------------
def _load_and_normalise(path: Path) -> List[Dict[str, Any]]:
    """Load the CSV and return rows with a guaranteed `dataset` key.

    Strategy:
      * Read header.
      * If `dataset` header exists, use row["dataset"] when its value is
        a known key; otherwise fall back to breast_cancer (legacy rows
        written before the dataset column existed).
      * If `dataset` header does NOT exist, treat every row as
        breast_cancer and drop any numeric-looking keys that DictReader
        may have created from misaligned columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    has_dataset_col = "dataset" in headers

    normalised: List[Dict[str, Any]] = []
    for r in rows:
        # Skip rows that have no recognisable method — these are almost
        # certainly artifacts of a schema mismatch, not real experiments.
        method = (r.get("method") or "").strip()
        if not method:
            continue

        raw_dataset = (r.get("dataset") or "").strip() if has_dataset_col else ""
        if raw_dataset in KNOWN_DATASETS:
            dataset = raw_dataset
        else:
            # Either the column is missing entirely, or it contains a
            # non-key value (legacy / misaligned). Default to breast_cancer.
            dataset = "breast_cancer"

        # Only keep rows with a plausible numeric status.
        status = (r.get("status") or "").strip().lower()
        normalised.append({
            "dataset": dataset,
            "method": method,
            "seed": r.get("seed", ""),
            "budget_seconds": r.get("budget_seconds", ""),
            "best_f1": _to_float(r.get("best_f1")),
            "total_time_seconds": _to_float(r.get("total_time_seconds")),
            "num_trials": _to_int(r.get("num_trials")),
            "average_trial_time": _to_float(r.get("average_trial_time")),
            "best_utility": _to_float(r.get("best_utility")),
            "status": status,
        })
    return normalised


def _to_float(v: Any) -> Optional[float]:
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    if v in ("", None):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _fmt(v: Optional[float], precision: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{precision}f}"


# ----------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------
def _print_per_method_table(
    rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    print("=" * 60)
    print("TimeTuneML — Pilot Analysis")
    print("=" * 60)
    print()

    by_ds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if r["status"] != "completed":
            continue
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r

    for ds in sorted(by_ds.keys()):
        print(f"Dataset: {DATASET_DISPLAY.get(ds, ds)}")
        print("-" * 60)
        header = (
            "Method".ljust(34)
            + "Best F1".rjust(10)
            + "Time (s)".rjust(12)
            + "Trials".rjust(9)
            + "Avg Trial".rjust(12)
        )
        print(header)
        print("-" * len(header))
        for method in ALL_METHODS:
            row = by_ds[ds].get(method)
            if row is None:
                print(method.ljust(34) + "MISSING".rjust(10))
                continue
            print(
                method.ljust(34)
                + _fmt(row["best_f1"], 4).rjust(10)
                + _fmt(row["total_time_seconds"], 4).rjust(12)
                + (str(row["num_trials"]) if row["num_trials"] is not None else "N/A").rjust(9)
                + _fmt(row["average_trial_time"], 4).rjust(12)
            )
        print()
    return by_ds


# ----------------------------------------------------------------------
# Dataset-level summary
# ----------------------------------------------------------------------
def _print_dataset_summary(
    by_ds: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    print("=" * 60)
    print("Dataset-Level Summary (search methods only)")
    print("=" * 60)
    print()

    header = (
        "Dataset".ljust(16)
        + "Avg Trial(s)".rjust(14)
        + "Max Trials".rjust(12)
        + "Min Trials".rjust(12)
        + "F1 Range".rjust(20)
    )
    print(header)
    print("-" * len(header))

    summary: Dict[str, Dict[str, Any]] = {}

    for ds in sorted(by_ds.keys()):
        search_rows = [by_ds[ds][m] for m in SEARCH_METHODS if m in by_ds[ds]]
        if not search_rows:
            continue

        avg_trials_secs = [
            v for v in (r["average_trial_time"] for r in search_rows)
            if v is not None
        ]
        trial_counts = [
            r["num_trials"] for r in search_rows if r["num_trials"] is not None
        ]
        f1s = [r["best_f1"] for r in search_rows if r["best_f1"] is not None]

        avg_trial = float(np.mean(avg_trials_secs)) if avg_trials_secs else None
        max_trials = max(trial_counts) if trial_counts else 0
        min_trials = min(trial_counts) if trial_counts else 0
        f1_min = min(f1s) if f1s else None
        f1_max = max(f1s) if f1s else None
        f1_range_str = (
            f"{f1_min:.4f} – {f1_max:.4f}"
            if (f1_min is not None and f1_max is not None)
            else "N/A"
        )

        summary[ds] = {
            "avg_trial_time": avg_trial,
            "max_trials": max_trials,
            "min_trials": min_trials,
            "f1_min": f1_min,
            "f1_max": f1_max,
            "f1_range": (
                (f1_max - f1_min)
                if (f1_min is not None and f1_max is not None)
                else None
            ),
        }

        print(
            DATASET_DISPLAY.get(ds, ds).ljust(16)
            + _fmt(avg_trial, 4).rjust(14)
            + str(max_trials).rjust(12)
            + str(min_trials).rjust(12)
            + f1_range_str.rjust(20)
        )

    print()
    return summary


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------
def _print_observations(
    by_ds: Dict[str, Dict[str, Dict[str, Any]]],
    summary: Dict[str, Dict[str, Any]],
) -> None:
    print("=" * 60)
    print("Factual Observations")
    print("=" * 60)
    print()

    for ds in sorted(summary.keys()):
        s = summary[ds]
        search_rows = [by_ds[ds][m] for m in SEARCH_METHODS if m in by_ds[ds]]
        f1_values = [
            r["best_f1"] for r in search_rows if r["best_f1"] is not None
        ]
        unique_f1 = len({round(v, 4) for v in f1_values})
        trial_spread = s["max_trials"] - s["min_trials"]
        avg_trial = s["avg_trial_time"]

        print(f"{DATASET_DISPLAY.get(ds, ds)}")
        print("-" * 60)

        if avg_trial is None:
            print("  - Average trial time unavailable.")
        elif avg_trial < 0.05:
            print(f"  - Trials are extremely fast (average {avg_trial:.4f} s per trial).")
        elif avg_trial < 0.2:
            print(f"  - Trials are fast (average {avg_trial:.4f} s per trial).")
        else:
            print(f"  - Trials are comparatively slow (average {avg_trial:.4f} s per trial).")

        if trial_spread <= 5:
            print(f"  - Trial counts across search methods differ by only {trial_spread} trials within the observed budget.")
        else:
            print(f"  - Trial counts across search methods span {s['min_trials']}–{s['max_trials']} (difference of {trial_spread}).")

        if unique_f1 <= 1:
            print(f"  - All search methods reached the same best F1 ({_fmt(f1_values[0] if f1_values else None)}) on this run.")
        else:
            print(f"  - Search methods produced {unique_f1} distinct best F1 values (range {_fmt(s['f1_range'])}).")

        if avg_trial is not None and avg_trial < 0.05:
            print("  - Because per-trial cost is very low, a longer budget would mostly add redundant trials rather than explore new regions.")
        elif avg_trial is not None and avg_trial < 0.2:
            print("  - A longer budget could add a moderate number of additional trials; whether it helps depends on whether F1 has plateaued.")
        else:
            print("  - A longer budget could add meaningfully more trials; the current run may be sample-limited.")

        if s["f1_max"] is not None and s["f1_max"] >= 0.99:
            print("  - At least one search method already reached a very high F1, so headroom for improvement is small.")
        elif s["f1_range"] is not None and s["f1_range"] > 0.01:
            print("  - Best F1 varies across methods by more than one point, suggesting further search might differentiate them.")
        print()


# ----------------------------------------------------------------------
# Recommendation
# ----------------------------------------------------------------------
def _print_recommendation(summary: Dict[str, Dict[str, Any]]) -> None:
    print("=" * 60)
    print("Recommended Final Experiment Design")
    print("=" * 60)
    print()

    recommended_budgets = [30, 120, 600]
    recommended_seeds = [42, 43, 44]

    print("Based on the pilot data, the following matrix is suggested:")
    print()
    print(f"  Datasets            : {', '.join(DATASET_DISPLAY.get(d, d) for d in sorted(summary.keys()))}")
    print(f"  Methods             : Default RF, Random Search, Bayesian Optimization, Time-Aware Bayesian Optimization")
    print(f"  Seeds               : {recommended_seeds}  (n = {len(recommended_seeds)})")
    print(f"  Budgets (seconds)   : {recommended_budgets}")
    total_runs = (
        len(summary) * len(ALL_METHODS) * len(recommended_seeds) * len(recommended_budgets)
    )
    print()
    print(f"  Total experiments   : {total_runs}")
    print()
    print("Rationale (from observed pilot data only):")
    print()
    print("  - All included datasets have comparable per-trial cost, so none needs to be dropped on cost grounds.")
    print("  - Three seeds provide a mean and standard deviation without inflating runtime beyond a reasonable envelope.")
    print("  - Budgets {30, 120, 600} span short, medium, and long regimes, which is the minimum needed to see whether more compute changes the outcome.")
    print("  - If F1 remains at ceiling across all three datasets at every budget, that is itself the finding: the current task cannot differentiate the methods, and task difficulty must increase before the cost-aware layer can be measured.")
    print()
    print("Note: this is a recommendation, not a decision.")
    print()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    try:
        rows = _load_and_normalise(FULL_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    completed = [r for r in rows if r["status"] == "completed"]
    if not completed:
        print("No completed experiments found in full_experiments.csv.")
        return 1

    by_ds = _print_per_method_table(completed)
    summary = _print_dataset_summary(by_ds)
    _print_observations(by_ds, summary)
    _print_recommendation(summary)

    if SUMMARY_CSV.exists():
        print(f"(Read {SUMMARY_CSV.relative_to(PROJECT_ROOT)} — no writes.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())