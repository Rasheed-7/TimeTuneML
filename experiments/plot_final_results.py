"""Generate report-quality figures for the TimeTuneML final results.

Reads only the existing CSVs:
    results/tables/full_experiments.csv
    results/tables/final_analysis.csv

Writes six figures to results/figures/ and a summary to
results/tables/final_figure_summary.md.

This script runs no experiments and modifies no result data.

Usage
-----
    python experiments/plot_final_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FULL_CSV = RESULTS_DIR / "full_experiments.csv"
ANALYSIS_CSV = RESULTS_DIR / "final_analysis.csv"
SUMMARY_MD = RESULTS_DIR / "final_figure_summary.md"

EXPECTED_DATASETS = ["breast_cancer", "iris", "wine"]
EXPECTED_BUDGETS = [30, 120, 600]
EXPECTED_METHODS = [
    "Default RF",
    "Random Search",
    "Bayesian Optimization",
    "Time-Aware Bayesian Optimization",
]

DATASET_DISPLAY = {
    "breast_cancer": "Breast Cancer",
    "iris": "Iris",
    "wine": "Wine",
}

# A short, method-name -> matplotlib line style/colour mapping. We do
# NOT hard-code specific colours for specific methods; matplotlib's
# default colour cycle is used, and each method gets a consistent marker
# so the lines are distinguishable in black-and-white printouts.
METHOD_MARKERS = {
    "Default RF": "o",
    "Random Search": "s",
    "Bayesian Optimization": "^",
    "Time-Aware Bayesian Optimization": "D",
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> Optional[float]:
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _group_stats(
    rows: List[Dict[str, Any]],
    value_key: str,
) -> Dict[Tuple[str, int, str], Dict[str, Optional[float]]]:
    """Group completed rows by (dataset, budget, method) and compute mean/std."""
    groups: Dict[Tuple[str, int, str], List[float]] = {}
    for r in rows:
        if (r.get("status") or "").lower() != "completed":
            continue
        val = _to_float(r.get(value_key))
        if val is None:
            continue
        try:
            key = (r["dataset"], int(float(r["budget_seconds"])), r["method"])
        except (KeyError, ValueError, TypeError):
            continue
        groups.setdefault(key, []).append(val)

    out: Dict[Tuple[str, int, str], Dict[str, Optional[float]]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
            "n": len(vals),
        }
    return out


# ----------------------------------------------------------------------
# Shared plotting helpers
# ----------------------------------------------------------------------
def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def _plot_metric_by_budget(
    stats: Dict[Tuple[str, int, str], Dict[str, Optional[float]]],
    value_label: str,
    title: str,
    out_path: Path,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    """One subplot per dataset; one line per method; x = configured budget."""
    fig, axes = plt.subplots(
        1, len(EXPECTED_DATASETS),
        figsize=(5.2 * len(EXPECTED_DATASETS), 4.4),
        sharey=False,
    )
    if len(EXPECTED_DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, EXPECTED_DATASETS):
        for method in EXPECTED_METHODS:
            xs: List[int] = []
            ys: List[float] = []
            es: List[float] = []
            for budget in EXPECTED_BUDGETS:
                info = stats.get((dataset, budget, method))
                if info is None or info["mean"] is None:
                    continue
                xs.append(budget)
                ys.append(info["mean"])
                es.append(info["std"] or 0.0)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=es,
                marker=METHOD_MARKERS.get(method, "o"),
                capsize=4,
                linewidth=1.6,
                markersize=6,
                label=method,
            )
        _style_axes(
            ax,
            DATASET_DISPLAY[dataset],
            "Configured budget (seconds)",
            value_label,
        )
        ax.set_xticks(EXPECTED_BUDGETS)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 1 — best F1 vs budget
# ----------------------------------------------------------------------
def fig_best_f1_vs_budget(rows: List[Dict[str, Any]]) -> Path:
    stats = _group_stats(rows, "best_f1")
    out = FIGURES_DIR / "best_f1_vs_budget.png"
    _plot_metric_by_budget(
        stats,
        value_label="Mean Best F1",
        title="Mean Best F1 by Configured Budget",
        out_path=out,
    )
    return out


# ----------------------------------------------------------------------
# Figure 2 — trials vs budget
# ----------------------------------------------------------------------
def fig_trials_vs_budget(rows: List[Dict[str, Any]]) -> Path:
    stats = _group_stats(rows, "num_trials")
    out = FIGURES_DIR / "trials_vs_budget.png"
    _plot_metric_by_budget(
        stats,
        value_label="Mean number of trials",
        title="Mean Number of Trials by Configured Budget",
        out_path=out,
    )
    return out


# ----------------------------------------------------------------------
# Figure 3 — runtime vs budget
# ----------------------------------------------------------------------
def fig_runtime_vs_budget(rows: List[Dict[str, Any]]) -> Path:
    stats = _group_stats(rows, "total_time_seconds")
    out = FIGURES_DIR / "runtime_vs_budget.png"
    _plot_metric_by_budget(
        stats,
        value_label="Mean actual runtime (seconds)",
        title="Mean Actual Runtime by Configured Budget",
        out_path=out,
    )
    return out


# ----------------------------------------------------------------------
# Figure 4 — Bayesian vs Time-Aware Bayesian
# ----------------------------------------------------------------------
def fig_bayesian_vs_timeaware(rows: List[Dict[str, Any]]) -> Path:
    """Two lines per dataset: standard Bayesian and Time-Aware Bayesian."""
    stats = _group_stats(rows, "best_f1")
    out = FIGURES_DIR / "bayesian_vs_timeaware.png"

    fig, axes = plt.subplots(
        1, len(EXPECTED_DATASETS),
        figsize=(5.2 * len(EXPECTED_DATASETS), 4.4),
        sharey=False,
    )
    if len(EXPECTED_DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, EXPECTED_DATASETS):
        for method, style in [
            ("Bayesian Optimization", {"marker": "^", "linestyle": "-"}),
            ("Time-Aware Bayesian Optimization", {"marker": "D", "linestyle": "--"}),
        ]:
            xs: List[int] = []
            ys: List[float] = []
            es: List[float] = []
            for budget in EXPECTED_BUDGETS:
                info = stats.get((dataset, budget, method))
                if info is None or info["mean"] is None:
                    continue
                xs.append(budget)
                ys.append(info["mean"])
                es.append(info["std"] or 0.0)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=es, capsize=4,
                linewidth=1.6, markersize=6, label=method,
                **style,
            )
        _style_axes(
            ax,
            DATASET_DISPLAY[dataset],
            "Configured budget (seconds)",
            "Mean Best F1",
        )
        ax.set_xticks(EXPECTED_BUDGETS)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Standard Bayesian vs Time-Aware Bayesian — Mean Best F1",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 5 — seed variability (mean ± std)
# ----------------------------------------------------------------------
def fig_f1_seed_variability(rows: List[Dict[str, Any]]) -> Path:
    """Bar chart of mean ± std F1 per (dataset, method, budget).

    Kept readable by plotting one subplot per dataset, with methods on
    the x-axis and a small horizontal offset per budget.
    """
    stats = _group_stats(rows, "best_f1")
    out = FIGURES_DIR / "f1_seed_variability.png"

    fig, axes = plt.subplots(
        1, len(EXPECTED_DATASETS),
        figsize=(5.8 * len(EXPECTED_DATASETS), 4.6),
        sharey=True,
    )
    if len(EXPECTED_DATASETS) == 1:
        axes = [axes]

    n_budgets = len(EXPECTED_BUDGETS)
    width = 0.8 / n_budgets  # total width per method group

    for ax, dataset in zip(axes, EXPECTED_DATASETS):
        method_positions = np.arange(len(EXPECTED_METHODS))
        for i, budget in enumerate(EXPECTED_BUDGETS):
            xs: List[float] = []
            means: List[float] = []
            errs: List[float] = []
            for j, method in enumerate(EXPECTED_METHODS):
                info = stats.get((dataset, budget, method))
                if info is None or info["mean"] is None:
                    continue
                xs.append(j + (i - (n_budgets - 1) / 2) * width)
                means.append(info["mean"])
                errs.append(info["std"] or 0.0)
            ax.bar(
                xs, means, width=width * 0.9,
                yerr=errs, capsize=3,
                label=f"{budget}s",
            )
        ax.set_xticks(method_positions)
        ax.set_xticklabels(
            [m.replace(" ", "\n") for m in EXPECTED_METHODS],
            fontsize=7,
        )
        _style_axes(
            ax,
            DATASET_DISPLAY[dataset],
            "Method",
            "Mean Best F1 (± std across seeds)",
        )
        ax.legend(title="Budget", fontsize=8)

    fig.suptitle(
        "Seed Variability of Best F1 (mean ± std over seeds 42/43/44)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 6 — time to target (only if populated)
# ----------------------------------------------------------------------
def fig_time_to_target(rows: List[Dict[str, Any]]) -> Optional[Path]:
    """Generate the figure only when time_to_target values exist."""
    completed = [r for r in rows if (r.get("status") or "").lower() == "completed"]
    ttt_rows = [r for r in completed if _to_float(r.get("time_to_target")) is not None]
    if not ttt_rows:
        return None

    out = FIGURES_DIR / "time_to_target.png"

    fig, axes = plt.subplots(
        1, len(EXPECTED_DATASETS),
        figsize=(5.2 * len(EXPECTED_DATASETS), 4.4),
        sharey=False,
    )
    if len(EXPECTED_DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, EXPECTED_DATASETS):
        for method in EXPECTED_METHODS:
            xs: List[int] = []
            ys: List[float] = []
            for budget in EXPECTED_BUDGETS:
                vals = [
                    _to_float(r.get("time_to_target"))
                    for r in ttt_rows
                    if r.get("dataset") == dataset
                    and r.get("method") == method
                    and int(float(r.get("budget_seconds", 0))) == budget
                ]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                xs.append(budget)
                ys.append(float(np.mean(vals)))
            if not xs:
                continue
            ax.plot(
                xs, ys,
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=1.6, markersize=6, label=method,
            )
        _style_axes(
            ax,
            DATASET_DISPLAY[dataset],
            "Configured budget (seconds)",
            "Mean time-to-target (seconds)",
        )
        ax.set_xticks(EXPECTED_BUDGETS)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Time to Reach Best-Observed F1 Target (per dataset)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Summary markdown
# ----------------------------------------------------------------------
def _write_summary(generated: List[Tuple[Path, str, str]], out_path: Path) -> None:
    lines = []
    lines.append("# TimeTuneML — Final Figure Summary")
    lines.append("")
    lines.append("Generated from `results/tables/full_experiments.csv`. "
                 "No experimental data was modified.")
    lines.append("")
    lines.append("| Figure | What it shows | Dimensions represented |")
    lines.append("| --- | --- | --- |")
    for path, description, dims in generated:
        lines.append(f"| `{path.name}` | {description} | {dims} |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("TimeTuneML — Final Figure Generation")
    print("=" * 60)
    print()

    rows = _load_csv(FULL_CSV)
    print(f"Loaded {len(rows)} rows from "
          f"{FULL_CSV.relative_to(PROJECT_ROOT)}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    generated: List[Tuple[Path, str, str]] = []

    p1 = fig_best_f1_vs_budget(rows)
    generated.append((
        p1,
        "Mean Best F1 vs configured budget; one line per method.",
        "dataset × method × budget",
    ))
    print(f"Wrote {p1.relative_to(PROJECT_ROOT)}")

    p2 = fig_trials_vs_budget(rows)
    generated.append((
        p2,
        "Mean number of trials vs configured budget; one line per method.",
        "dataset × method × budget",
    ))
    print(f"Wrote {p2.relative_to(PROJECT_ROOT)}")

    p3 = fig_runtime_vs_budget(rows)
    generated.append((
        p3,
        "Mean actual runtime vs configured budget; one line per method.",
        "dataset × method × budget",
    ))
    print(f"Wrote {p3.relative_to(PROJECT_ROOT)}")

    p4 = fig_bayesian_vs_timeaware(rows)
    generated.append((
        p4,
        "Standard Bayesian vs Time-Aware Bayesian mean Best F1 by budget.",
        "dataset × {Bayesian, Time-Aware Bayesian} × budget",
    ))
    print(f"Wrote {p4.relative_to(PROJECT_ROOT)}")

    p5 = fig_f1_seed_variability(rows)
    generated.append((
        p5,
        "Mean ± std Best F1 across seeds, grouped by method and budget.",
        "dataset × method × budget × seed",
    ))
    print(f"Wrote {p5.relative_to(PROJECT_ROOT)}")

    p6 = fig_time_to_target(rows)
    if p6 is not None:
        generated.append((
            p6,
            "Mean time-to-target vs configured budget; one line per method.",
            "dataset × method × budget",
        ))
        print(f"Wrote {p6.relative_to(PROJECT_ROOT)}")
    else:
        print("Skipped time_to_target.png — no time_to_target values in CSV.")

    _write_summary(generated, SUMMARY_MD)
    print(f"Wrote {SUMMARY_MD.relative_to(PROJECT_ROOT)}")

    print()
    print("=" * 60)
    print("Generated Files")
    print("=" * 60)
    for path, _, _ in generated:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(f"  {SUMMARY_MD.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Total figures written: {len(generated)}")
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())