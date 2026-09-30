"""Run the TimeTuneML benchmark across all four methods.

Usage
-----
    python experiments/run_benchmark.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow running as a plain script: add `src/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timetuneml.experiments.benchmark import run_all

SEED = 42
BUDGET_SECONDS = 30.0
WARMUP_TRIALS = 5

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
CSV_PATH = RESULTS_DIR / "benchmark_30s_seed42.csv"

CSV_COLUMNS = [
    "method",
    "seed",
    "budget_seconds",
    "best_f1",
    "total_time_seconds",
    "num_trials",
    "best_parameters",
    "best_utility",
    "average_trial_time",
]


def _fmt(value, width: int = 10, precision: int = 4) -> str:
    if value is None:
        return "N/A".ljust(width)
    if isinstance(value, float):
        return f"{value:.{precision}f}".ljust(width)
    return str(value).ljust(width)


def main() -> None:
    print("=" * 60)
    print("TimeTuneML — Benchmark")
    print("=" * 60)
    print()
    print("Dataset : Breast Cancer")
    print("Model   : Random Forest")
    print("Metric  : F1")
    print(f"Seed    : {SEED}")
    print(f"Budget  : {BUDGET_SECONDS:.0f} seconds")
    print()

    print("Running Default Baseline...")
    print("Running Random Search...")
    print("Running Bayesian Optimization...")
    print("Running Time-Aware Bayesian Optimization...")
    print()

    results = run_all(
        seed=SEED,
        budget_seconds=BUDGET_SECONDS,
        warmup_trials=WARMUP_TRIALS,
    )

    # ---- print table ---------------------------------------------------
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print()
    header = (
        "Method".ljust(30)
        + "Best F1".ljust(12)
        + "Time (s)".ljust(12)
        + "Trials".ljust(10)
    )
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            r.method.ljust(30)
            + _fmt(r.best_f1, width=12)
            + _fmt(r.total_time_seconds, width=12)
            + str(r.num_trials).ljust(10)
        )
        print(row)
    print()

    # ---- write CSV -----------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())

    print("Results saved to:")
    print(str(CSV_PATH.relative_to(PROJECT_ROOT)))
    print()
    print("Done.")


if __name__ == "__main__":
    main()