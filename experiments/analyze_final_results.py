"""Analyze the completed TimeTuneML 108-experiment matrix.

Reads only the existing CSVs:
    results/tables/full_experiments.csv
    results/tables/experiment_summary.csv

Produces:
    results/tables/final_analysis.csv
    results/tables/final_analysis.md

Prints a concise terminal summary.

This script performs no experiments, modifies no results, and makes no
claims about method superiority. All output is descriptive.

Usage
-----
    python experiments/analyze_final_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FULL_CSV = RESULTS_DIR / "full_experiments.csv"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
ANALYSIS_CSV = RESULTS_DIR / "final_analysis.csv"
ANALYSIS_MD = RESULTS_DIR / "final_analysis.md"

# Expected matrix (dataset, method, seed, budget).
EXPECTED_DATASETS = ["breast_cancer", "iris", "wine"]
EXPECTED_METHODS = [
    "Default RF",
    "Random Search",
    "Bayesian Optimization",
    "Time-Aware Bayesian Optimization",
]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_BUDGETS = [30, 120, 600]

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


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(row: Dict[str, Any], key: str) -> Optional[int]:
    v = _f(row, key)
    return int(v) if v is not None else None


def _key(row: Dict[str, Any]) -> Tuple[str, str, int, int]:
    return (
        row.get("dataset", ""),
        row.get("method", ""),
        int(float(row.get("seed", 0))),
        int(float(row.get("budget_seconds", 0))),
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _validate_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confirm the expected 108-combination matrix is present exactly once."""
    report: Dict[str, Any] = {
        "expected": len(EXPECTED_DATASETS)
        * len(EXPECTED_METHODS)
        * len(EXPECTED_SEEDS)
        * len(EXPECTED_BUDGETS),
        "found_rows": len(rows),
        "completed": 0,
        "failed": 0,
        "other": 0,
        "duplicates": [],
        "missing": [],
        "unexpected": [],
    }

    seen: Dict[Tuple[str, str, int, int], int] = {}
    for r in rows:
        status = (r.get("status") or "").strip().lower()
        if status == "completed":
            report["completed"] += 1
        elif status == "failed":
            report["failed"] += 1
        else:
            report["other"] += 1

        k = _key(r)
        seen[k] = seen.get(k, 0) + 1

    report["duplicates"] = [k for k, n in seen.items() if n > 1]

    expected_set = {
        (d, m, s, b)
        for d in EXPECTED_DATASETS
        for m in EXPECTED_METHODS
        for s in EXPECTED_SEEDS
        for b in EXPECTED_BUDGETS
    }
    report["missing"] = sorted(expected_set - set(seen.keys()))
    report["unexpected"] = sorted(set(seen.keys()) - expected_set)
    return report


# ----------------------------------------------------------------------
# Grouping helpers
# ----------------------------------------------------------------------
def _group_completed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if (r.get("status") or "").lower() == "completed"]


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if len(arr) > 1 else None,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _by_dataset_budget(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Group by (dataset, budget, method)."""
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        try:
            key = (r["dataset"], int(float(r["budget_seconds"])), r["method"])
        except (KeyError, ValueError, TypeError):
            continue
        groups.setdefault(key, []).append(r)
    return {k: {"rows": v, **_aggregate(v)} for k, v in groups.items()}


def _aggregate(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    f1s = [v for v in (_f(r, "best_f1") for r in group) if v is not None]
    times = [v for v in (_f(r, "total_time_seconds") for r in group) if v is not None]
    trials = [v for v in (_f(r, "num_trials") for r in group) if v is not None]
    avg_trial = [v for v in (_f(r, "average_trial_time") for r in group) if v is not None]
    return {
        "n_runs": len(group),
        "f1": _stats(f1s),
        "runtime": _stats(times),
        "trials": _stats(trials),
        "avg_trial": _stats(avg_trial),
    }


# ----------------------------------------------------------------------
# Rendering — terminal
# ----------------------------------------------------------------------
def _fmt(v: Optional[float], p: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{p}f}"


def _print_validation(report: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Matrix Validation")
    print("=" * 60)
    print(f"Expected combinations : {report['expected']}")
    print(f"CSV rows              : {report['found_rows']}")
    print(f"Completed             : {report['completed']}")
    print(f"Failed                : {report['failed']}")
    print(f"Other statuses        : {report['other']}")
    print(f"Duplicate keys        : {len(report['duplicates'])}")
    print(f"Missing combinations  : {len(report['missing'])}")
    print(f"Unexpected keys       : {len(report['unexpected'])}")
    if report["duplicates"]:
        for k in report["duplicates"]:
            print(f"  DUP  {k}")
    if report["missing"]:
        for k in report["missing"][:20]:
            print(f"  MISS {k}")
        if len(report["missing"]) > 20:
            print(f"  ... and {len(report['missing']) - 20} more")
    if report["unexpected"]:
        for k in report["unexpected"][:20]:
            print(f"  UNEXP {k}")
    print()


def _print_grouped_by_dataset_budget(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> None:
    print("=" * 60)
    print("Dataset x Budget x Method — F1 statistics (mean ± std, n=3 seeds)")
    print("=" * 60)
    for dataset in EXPECTED_DATASETS:
        for budget in EXPECTED_BUDGETS:
            print()
            print(f"{DATASET_DISPLAY[dataset]} — {budget}s budget")
            print("-" * 60)
            header = (
                "Method".ljust(34)
                + "mean F1".rjust(10)
                + "std".rjust(9)
                + "median".rjust(9)
                + "mean trials".rjust(13)
                + "mean runtime".rjust(14)
            )
            print(header)
            print("-" * len(header))
            for method in EXPECTED_METHODS:
                key = (dataset, budget, method)
                info = grouped.get(key)
                if info is None:
                    print(method.ljust(34) + "MISSING")
                    continue
                f1 = info["f1"]
                trials = info["trials"]
                runtime = info["runtime"]
                print(
                    method.ljust(34)
                    + _fmt(f1["mean"]).rjust(10)
                    + _fmt(f1["std"]).rjust(9)
                    + _fmt(f1["median"]).rjust(9)
                    + _fmt(trials["mean"], 1).rjust(13)
                    + _fmt(runtime["mean"], 2).rjust(14)
                )
        print()


def _print_grouped_by_method_budget(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> None:
    print("=" * 60)
    print("Method x Budget — aggregated across datasets (n=9 runs each)")
    print("=" * 60)
    for budget in EXPECTED_BUDGETS:
        print()
        print(f"Budget: {budget}s")
        print("-" * 60)
        header = (
            "Method".ljust(34)
            + "mean F1".rjust(10)
            + "std F1".rjust(9)
            + "mean runtime".rjust(14)
            + "mean trials".rjust(13)
        )
        print(header)
        print("-" * len(header))
        for method in EXPECTED_METHODS:
            f1s: List[float] = []
            runtimes: List[float] = []
            trials: List[float] = []
            for dataset in EXPECTED_DATASETS:
                info = grouped.get((dataset, budget, method))
                if info is None:
                    continue
                if info["f1"]["mean"] is not None:
                    f1s.extend([r for r in (_f(r, "best_f1") for r in info["rows"]) if r is not None])
                if info["runtime"]["mean"] is not None:
                    runtimes.extend([r for r in (_f(r, "total_time_seconds") for r in info["rows"]) if r is not None])
                if info["trials"]["mean"] is not None:
                    trials.extend([r for r in (_f(r, "num_trials") for r in info["rows"]) if r is not None])
            s_f1 = _stats(f1s)
            s_run = _stats(runtimes)
            s_trials = _stats(trials)
            print(
                method.ljust(34)
                + _fmt(s_f1["mean"]).rjust(10)
                + _fmt(s_f1["std"]).rjust(9)
                + _fmt(s_run["mean"], 2).rjust(14)
                + _fmt(s_trials["mean"], 1).rjust(13)
            )
    print()


def _print_budget_effect(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> None:
    print("=" * 60)
    print("Budget Effect — 30s -> 120s -> 600s (mean F1 and mean trials)")
    print("=" * 60)
    for dataset in EXPECTED_DATASETS:
        print()
        print(f"{DATASET_DISPLAY[dataset]}")
        print("-" * 60)
        header = (
            "Method".ljust(34)
            + "F1 30s".rjust(9)
            + "F1 120s".rjust(9)
            + "F1 600s".rjust(9)
            + "Tr 30s".rjust(9)
            + "Tr 120s".rjust(9)
            + "Tr 600s".rjust(9)
        )
        print(header)
        print("-" * len(header))
        for method in EXPECTED_METHODS:
            f1_vals: List[Optional[float]] = []
            tr_vals: List[Optional[float]] = []
            for budget in EXPECTED_BUDGETS:
                info = grouped.get((dataset, budget, method))
                f1_vals.append(info["f1"]["mean"] if info else None)
                tr_vals.append(info["trials"]["mean"] if info else None)
            print(
                method.ljust(34)
                + _fmt(f1_vals[0]).rjust(9)
                + _fmt(f1_vals[1]).rjust(9)
                + _fmt(f1_vals[2]).rjust(9)
                + _fmt(tr_vals[0], 1).rjust(9)
                + _fmt(tr_vals[1], 1).rjust(9)
                + _fmt(tr_vals[2], 1).rjust(9)
            )
    print()


def _print_seed_variability(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> None:
    print("=" * 60)
    print("Seed Variability — min / mean / max F1 across seeds 42,43,44")
    print("=" * 60)
    for dataset in EXPECTED_DATASETS:
        print()
        print(f"{DATASET_DISPLAY[dataset]}")
        print("-" * 60)
        header = (
            "Method".ljust(34)
            + "Budget".rjust(8)
            + "min".rjust(9)
            + "mean".rjust(9)
            + "max".rjust(9)
            + "std".rjust(9)
        )
        print(header)
        print("-" * len(header))
        for method in EXPECTED_METHODS:
            for budget in EXPECTED_BUDGETS:
                info = grouped.get((dataset, budget, method))
                if info is None:
                    continue
                f1 = info["f1"]
                print(
                    method.ljust(34)
                    + str(budget).rjust(8)
                    + _fmt(f1["min"]).rjust(9)
                    + _fmt(f1["mean"]).rjust(9)
                    + _fmt(f1["max"]).rjust(9)
                    + _fmt(f1["std"]).rjust(9)
                )
    print()


def _print_time_aware_vs_bayesian(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> None:
    print("=" * 60)
    print("Time-Aware Bayesian vs Bayesian — descriptive deltas")
    print("=" * 60)
    print("Positive F1 delta = Time-Aware observed higher mean F1.")
    print("Positive runtime delta = Time-Aware ran longer on average.")
    print()
    header = (
        "Dataset".ljust(15)
        + "Budget".rjust(8)
        + "Δ mean F1".rjust(12)
        + "Δ runtime".rjust(12)
        + "Δ trials".rjust(11)
    )
    print(header)
    print("-" * len(header))
    for dataset in EXPECTED_DATASETS:
        for budget in EXPECTED_BUDGETS:
            bay = grouped.get((dataset, budget, "Bayesian Optimization"))
            ta = grouped.get(
                (dataset, budget, "Time-Aware Bayesian Optimization")
            )
            if bay is None or ta is None:
                print(f"{DATASET_DISPLAY[dataset]:<15}{str(budget):>8}  MISSING")
                continue
            df1 = _delta(ta["f1"]["mean"], bay["f1"]["mean"])
            drun = _delta(ta["runtime"]["mean"], bay["runtime"]["mean"])
            dtr = _delta(ta["trials"]["mean"], bay["trials"]["mean"])
            print(
                DATASET_DISPLAY[dataset].ljust(15)
                + str(budget).rjust(8)
                + _fmt(df1).rjust(12)
                + _fmt(drun, 2).rjust(12)
                + _fmt(dtr, 1).rjust(11)
            )
    print()


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


# ----------------------------------------------------------------------
# Patterns (observations only)
# ----------------------------------------------------------------------
def _identify_patterns(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]]
) -> List[str]:
    out: List[str] = []

    # Whether longer budgets improve F1
    any_improvement = False
    any_flat = False
    for dataset in EXPECTED_DATASETS:
        for method in SEARCH_METHODS:
            f1_30 = grouped.get((dataset, 30, method))
            f1_600 = grouped.get((dataset, 600, method))
            if not f1_30 or not f1_600:
                continue
            m30 = f1_30["f1"]["mean"]
            m600 = f1_600["f1"]["mean"]
            if m30 is None or m600 is None:
                continue
            if m600 - m30 > 0.005:
                any_improvement = True
            elif abs(m600 - m30) <= 0.005:
                any_flat = True
    if any_improvement:
        out.append("On at least one dataset/method, mean F1 is higher at 600s than at 30s.")
    if any_flat:
        out.append("On at least one dataset/method, mean F1 at 600s is within 0.005 of mean F1 at 30s (plateau behaviour).")

    # Whether trial counts scale with budget
    scales = True
    for dataset in EXPECTED_DATASETS:
        for method in SEARCH_METHODS:
            t30 = grouped.get((dataset, 30, method))
            t600 = grouped.get((dataset, 600, method))
            if not t30 or not t600:
                continue
            m30 = t30["trials"]["mean"]
            m600 = t600["trials"]["mean"]
            if m30 is None or m600 is None:
                continue
            if m600 <= m30:
                scales = False
    if scales:
        out.append("For all search methods, mean trial count at 600s exceeds mean trial count at 30s.")

    # Whether Time-Aware differs from standard Bayesian
    ta_differs = False
    for dataset in EXPECTED_DATASETS:
        for budget in EXPECTED_BUDGETS:
            bay = grouped.get((dataset, budget, "Bayesian Optimization"))
            ta = grouped.get((dataset, budget, "Time-Aware Bayesian Optimization"))
            if not bay or not ta:
                continue
            df1 = _delta(ta["f1"]["mean"], bay["f1"]["mean"])
            dtr = _delta(ta["trials"]["mean"], bay["trials"]["mean"])
            if (df1 is not None and abs(df1) > 0.005) or (
                dtr is not None and abs(dtr) > 5
            ):
                ta_differs = True
                break
    out.append(
        "Time-Aware Bayesian and standard Bayesian differ by more than 0.005 mean F1 "
        "or more than 5 mean trials on at least one dataset/budget."
        if ta_differs
        else "Time-Aware Bayesian and standard Bayesian remain within 0.005 mean F1 "
             "and within 5 mean trials on every dataset/budget."
    )

    # Seed variability size
    all_stds: List[float] = []
    for info in grouped.values():
        s = info["f1"]["std"]
        if s is not None:
            all_stds.append(s)
    if all_stds:
        max_std = max(all_stds)
        out.append(
            f"Largest observed F1 standard deviation across seeds is {max_std:.4f}."
        )

    # Per-dataset behaviour
    for dataset in EXPECTED_DATASETS:
        f1s = [
            grouped[(dataset, b, m)]["f1"]["mean"]
            for b in EXPECTED_BUDGETS
            for m in SEARCH_METHODS
            if (dataset, b, m) in grouped
            and grouped[(dataset, b, m)]["f1"]["mean"] is not None
        ]
        if f1s:
            lo, hi = min(f1s), max(f1s)
            out.append(
                f"{DATASET_DISPLAY[dataset]}: search-method mean F1 ranges "
                f"from {lo:.4f} to {hi:.4f} across all budgets."
            )
    return out


# ----------------------------------------------------------------------
# Time-to-target analysis (best-effort)
# ----------------------------------------------------------------------
def _time_to_target_available(rows: List[Dict[str, Any]]) -> bool:
    """The stored CSV does not carry trial-level F1 trajectories.

    Only the summary fields (best_f1, total_time_seconds, num_trials) are
    persisted. Trial-level data would be needed to compute the earliest
    cumulative time at which best-so-far F1 crosses a target.
    """
    required = {"time_to_target"}
    return any(k in rows[0] for k in required) if rows else False


def _summarise_time_to_target(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort summary of the time_to_target column, if populated."""
    populated = [
        _f(r, "time_to_target")
        for r in rows
        if (r.get("status") or "").lower() == "completed"
        and _f(r, "time_to_target") is not None
    ]
    return {
        "n_rows_with_value": len(populated),
        "n_rows_total": len(rows),
        "values": populated,
    }


# ----------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------
ANALYSIS_COLUMNS = [
    "dataset",
    "method",
    "budget_seconds",
    "n_runs",
    "mean_best_f1",
    "std_best_f1",
    "median_best_f1",
    "min_best_f1",
    "max_best_f1",
    "mean_total_time_seconds",
    "std_total_time_seconds",
    "mean_num_trials",
    "std_num_trials",
    "mean_average_trial_time",
]


def _write_analysis_csv(
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        for (dataset, budget, method), info in sorted(grouped.items()):
            writer.writerow({
                "dataset": dataset,
                "method": method,
                "budget_seconds": budget,
                "n_runs": info["n_runs"],
                "mean_best_f1": info["f1"]["mean"],
                "std_best_f1": info["f1"]["std"],
                "median_best_f1": info["f1"]["median"],
                "min_best_f1": info["f1"]["min"],
                "max_best_f1": info["f1"]["max"],
                "mean_total_time_seconds": info["runtime"]["mean"],
                "std_total_time_seconds": info["runtime"]["std"],
                "mean_num_trials": info["trials"]["mean"],
                "std_num_trials": info["trials"]["std"],
                "mean_average_trial_time": info["avg_trial"]["mean"],
            })


# ----------------------------------------------------------------------
# Markdown output
# ----------------------------------------------------------------------
def _write_markdown(
    validation: Dict[str, Any],
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]],
    patterns: List[str],
    ttt_info: Dict[str, Any],
    out_path: Path,
) -> None:
    lines: List[str] = []

    lines.append("# TimeTuneML — Final Analysis")
    lines.append("")
    lines.append("Descriptive statistics only. No ranking or superiority claim is made.")
    lines.append("")

    # Validation
    lines.append("## Matrix Validation")
    lines.append("")
    lines.append(f"- Expected combinations : {validation['expected']}")
    lines.append(f"- CSV rows              : {validation['found_rows']}")
    lines.append(f"- Completed             : {validation['completed']}")
    lines.append(f"- Failed                : {validation['failed']}")
    lines.append(f"- Duplicate keys        : {len(validation['duplicates'])}")
    lines.append(f"- Missing combinations  : {len(validation['missing'])}")
    lines.append("")

    # Dataset x budget x method
    lines.append("## Results by Dataset, Budget, and Method")
    lines.append("")
    lines.append("| Dataset | Method | Budget (s) | n | mean F1 | std F1 | median F1 | mean trials | mean runtime (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (dataset, budget, method), info in sorted(grouped.items()):
        lines.append(
            "| {d} | {m} | {b} | {n} | {mf} | {sf} | {med} | {mt} | {mr} |".format(
                d=DATASET_DISPLAY.get(dataset, dataset),
                m=method,
                b=budget,
                n=info["n_runs"],
                mf=_fmt(info["f1"]["mean"]),
                sf=_fmt(info["f1"]["std"]),
                med=_fmt(info["f1"]["median"]),
                mt=_fmt(info["trials"]["mean"], 1),
                mr=_fmt(info["runtime"]["mean"], 2),
            )
        )
    lines.append("")

    # Time-aware vs Bayesian
    lines.append("## Time-Aware Bayesian vs Standard Bayesian (descriptive)")
    lines.append("")
    lines.append("| Dataset | Budget (s) | Δ mean F1 | Δ mean runtime (s) | Δ mean trials |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for dataset in EXPECTED_DATASETS:
        for budget in EXPECTED_BUDGETS:
            bay = grouped.get((dataset, budget, "Bayesian Optimization"))
            ta = grouped.get((dataset, budget, "Time-Aware Bayesian Optimization"))
            if bay is None or ta is None:
                continue
            df1 = _delta(ta["f1"]["mean"], bay["f1"]["mean"])
            drun = _delta(ta["runtime"]["mean"], bay["runtime"]["mean"])
            dtr = _delta(ta["trials"]["mean"], bay["trials"]["mean"])
            lines.append(
                "| {d} | {b} | {a} | {r} | {t} |".format(
                    d=DATASET_DISPLAY.get(dataset, dataset),
                    b=budget,
                    a=_fmt(df1),
                    r=_fmt(drun, 2),
                    t=_fmt(dtr, 1),
                )
            )
    lines.append("")

    # Time-to-target
    lines.append("## Time-to-Target")
    lines.append("")
    if ttt_info["n_rows_with_value"] == 0:
        lines.append(
            "Not calculable from the currently stored data. The saved CSV "
            "does not include trial-level F1 trajectories, which are "
            "required to identify the earliest cumulative time at which "
            "best-so-far F1 crosses a target."
        )
    else:
        lines.append(
            f"{ttt_info['n_rows_with_value']} of {ttt_info['n_rows_total']} "
            "rows carry a `time_to_target` value."
        )
    lines.append("")

    # Patterns
    lines.append("## Empirical Patterns")
    lines.append("")
    for p in patterns:
        lines.append(f"- {p}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("TimeTuneML — Final Analysis")
    print("=" * 60)
    print()

    rows = _load_rows(FULL_CSV)
    validation = _validate_matrix(rows)
    _print_validation(validation)

    completed = _group_completed(rows)
    grouped = _by_dataset_budget(completed)

    _print_grouped_by_dataset_budget(grouped)
    _print_grouped_by_method_budget(grouped)
    _print_budget_effect(grouped)
    _print_seed_variability(grouped)
    _print_time_aware_vs_bayesian(grouped)

    print("=" * 60)
    print("Time-to-Target")
    print("=" * 60)
    ttt_info = _summarise_time_to_target(rows)
    if ttt_info["n_rows_with_value"] == 0:
        print(
            "Not calculable from the currently stored data. Trial-level F1 "
            "trajectories are not persisted in the CSV; only per-experiment "
            "summary fields (best_f1, total_time_seconds, num_trials) are."
        )
    else:
        print(
            f"{ttt_info['n_rows_with_value']} of {ttt_info['n_rows_total']} "
            "rows carry a `time_to_target` value."
        )
    print()

    patterns = _identify_patterns(grouped)
    print("=" * 60)
    print("Empirical Patterns (observations only)")
    print("=" * 60)
    for p in patterns:
        print(f"  - {p}")
    print()

    _write_analysis_csv(grouped, ANALYSIS_CSV)
    _write_markdown(validation, grouped, patterns, ttt_info, ANALYSIS_MD)

    print("=" * 60)
    print("Files Written")
    print("=" * 60)
    print(f"CSV      : {ANALYSIS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Markdown : {ANALYSIS_MD.relative_to(PROJECT_ROOT)}")
    print()
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())