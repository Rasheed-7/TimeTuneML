"""Generate benchmark plots and a Markdown summary for TimeTuneML.

Reads the benchmark CSV produced by Milestone 6 and writes:

    results/figures/benchmark_best_f1.png
    results/figures/benchmark_runtime.png
    results/figures/benchmark_trials.png
    results/tables/benchmark_summary.md

Usage
-----
    python experiments/plot_benchmark.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script runs headless (CI, servers).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "tables" / "benchmark_30s_seed42.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_PATH = TABLES_DIR / "benchmark_summary.md"


# ----------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------
def load_benchmark_rows(csv_path: Path) -> list[dict]:
    """Load benchmark rows from the Milestone 6 CSV.

    Numeric fields that may be blank are converted to float or None.
    Returns a list of dicts in CSV order.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found at {csv_path}. "
            "Run `python experiments/run_benchmark.py` first."
        )

    rows: list[dict] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "method": raw["method"],
                "seed": int(raw["seed"]),
                "budget_seconds": float(raw["budget_seconds"]),
                "best_f1": _to_float_or_none(raw.get("best_f1")),
                "total_time_seconds": _to_float_or_none(raw.get("total_time_seconds")),
                "num_trials": int(raw["num_trials"]) if raw.get("num_trials") else 0,
                "best_parameters": raw.get("best_parameters", ""),
                "best_utility": _to_float_or_none(raw.get("best_utility")),
                "average_trial_time": _to_float_or_none(raw.get("average_trial_time")),
            }
            rows.append(row)
    return rows


def _to_float_or_none(value) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
    value_fmt: str = "{:.4f}",
) -> None:
    """Render a single bar chart and save it as PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="black")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Rotate labels a little for readability.
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Annotate each bar with its numeric value.
    for bar, value in zip(bars, values):
        ax.annotate(
            value_fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Give a little headroom so the annotations don't clip.
    if values:
        top = max(values)
        if top > 0:
            ax.set_ylim(0, top * 1.15)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------
def write_summary(rows: list[dict], out_path: Path) -> None:
    """Write a factual Markdown summary of the benchmark results."""
    # All rows share the same seed / budget in this milestone.
    seed = rows[0]["seed"] if rows else 42
    budget = rows[0]["budget_seconds"] if rows else 30.0

    lines: list[str] = []
    lines.append("# TimeTuneML Benchmark Summary")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append("- **Dataset:** Breast Cancer (sklearn `load_breast_cancer`)")
    lines.append("- **Model:** Random Forest Classifier")
    lines.append("- **Metric:** F1")
    lines.append(f"- **Seed:** {seed}")
    lines.append(f"- **Budget:** {budget:.0f} seconds")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Method | Best F1 | Total Time (s) | Number of Trials |")
    lines.append("| --- | ---: | ---: | ---: |")
    for r in rows:
        best_f1 = f"{r['best_f1']:.4f}" if r["best_f1"] is not None else "N/A"
        total_time = (
            f"{r['total_time_seconds']:.4f}"
            if r["total_time_seconds"] is not None
            else "N/A"
        )
        lines.append(
            f"| {r['method']} | {best_f1} | {total_time} | {r['num_trials']} |"
        )
    lines.append("")

    # Factual observations only — no rankings, no claims.
    lines.append("## Observations")
    lines.append("")
    lines.append(
        "- All four methods reached the same best F1 value on this single "
        "30-second, seed-42 run."
    )
    lines.append(
        "- The Default RF completed in substantially less time because it "
        "performed a single training run and no search."
    )
    lines.append(
        "- The three search-based methods each consumed approximately the "
        "configured 30-second budget."
    )
    lines.append(
        "- Within that budget the search methods evaluated different numbers "
        "of trials, reflecting differences in sampled hyperparameters and "
        "per-trial runtimes."
    )
    lines.append(
        "- These results describe one experiment only and should not be used "
        "to rank the methods; multi-seed and multi-budget runs are required "
        "before drawing general conclusions."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Benchmark Plots & Summary")
    print("=" * 60)
    print()

    rows = load_benchmark_rows(CSV_PATH)
    if not rows:
        print(f"No rows found in {CSV_PATH}.")
        sys.exit(1)

    methods = [r["method"] for r in rows]
    best_f1 = [r["best_f1"] if r["best_f1"] is not None else 0.0 for r in rows]
    runtimes = [
        r["total_time_seconds"] if r["total_time_seconds"] is not None else 0.0
        for r in rows
    ]
    trials = [float(r["num_trials"]) for r in rows]

    # ---- Plot 1: Best F1 ----------------------------------------------
    p1 = FIGURES_DIR / "benchmark_best_f1.png"
    _bar_chart(
        labels=methods,
        values=best_f1,
        title="Best F1 by Optimization Method",
        ylabel="Best F1",
        out_path=p1,
        value_fmt="{:.4f}",
    )
    print(f"Wrote {p1.relative_to(PROJECT_ROOT)}")

    # ---- Plot 2: Runtime ----------------------------------------------
    p2 = FIGURES_DIR / "benchmark_runtime.png"
    _bar_chart(
        labels=methods,
        values=runtimes,
        title="Total Runtime by Optimization Method",
        ylabel="Total Time (seconds)",
        out_path=p2,
        value_fmt="{:.2f}",
    )
    print(f"Wrote {p2.relative_to(PROJECT_ROOT)}")

    # ---- Plot 3: Trials -----------------------------------------------
    p3 = FIGURES_DIR / "benchmark_trials.png"
    _bar_chart(
        labels=methods,
        values=trials,
        title="Number of Trials by Optimization Method",
        ylabel="Number of Trials",
        out_path=p3,
        value_fmt="{:.0f}",
    )
    print(f"Wrote {p3.relative_to(PROJECT_ROOT)}")

    # ---- Markdown summary ---------------------------------------------
    write_summary(rows, SUMMARY_PATH)
    print(f"Wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("All outputs generated successfully.")
    print("Done.")


if __name__ == "__main__":
    main()