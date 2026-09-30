# TimeTuneML — Reproducibility Guide

This document describes how to reproduce the TimeTuneML 108-experiment matrix on a fresh machine.

---

## 1. Prerequisites

- **Python 3.10 or newer.** Earlier versions may work but are not tested.
- **Git** (to clone the repository).
- **A machine with at least 2 GB free RAM** and one CPU core.
- **~6 GB free disk space** (mostly for the virtual environment and cached packages).

The full matrix takes **approximately 5–6 hours of wall-clock time** on a modern laptop. Individual experiments can be run in isolation if you only want a subset.

---

## 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/TimeTuneML.git
cd TimeTuneML
```

Replace `YOUR_USERNAME` with your GitHub username. If you cloned via SSH instead of HTTPS, use the SSH URL.

---

## 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt once activated.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install numpy scikit-learn matplotlib optuna streamlit
```

Versions used during the original run:

- `numpy` ≥ 1.24
- `scikit-learn` ≥ 1.3
- `matplotlib` ≥ 3.7
- `optuna` ≥ 3.4
- `streamlit` ≥ 1.30

Exact minor versions may affect absolute timings slightly, but not the experimental design or the F1 values.

---

## 5. Verify the installation

Run the dataset sanity-check script:

```bash
python experiments/test_datasets.py
```

Expected output:

```
============================================================
TimeTuneML — Dataset Validation
============================================================

Dataset                Samples  Features   Classes
--------------------------------------------------
Breast Cancer              569        30         2
Iris                       150         4         3
Wine                       178        13         3

Sanity checks (split / preprocess / fit / F1)
------------------------------------------------------------
Breast Cancer        OK    F1=0.9655
Iris                 OK    F1=0.9xxx
Wine                 OK    F1=0.9xxx

All datasets passed the sanity check.
```

All three datasets should be marked `OK`.

Run the fast baseline:

```bash
python experiments/run_baseline.py
```

Expected: Breast Cancer baseline prints an F1 near `0.9655`.

---

## 6. Inspect the experiment matrix

```bash
python experiments/run_full_experiments.py --final --dry-run
```

Expected: prints the 108 tuples in order and ends with

```
Unique combinations: 108
Duplicates         : 0
```

This confirms the matrix is the expected `3 datasets × 4 methods × 3 seeds × 3 budgets` with no duplicates.

---

## 7. Run the full matrix

```bash
python experiments/run_full_experiments.py --final
```

**Runtime:** approximately 5–6 hours on a modern laptop. The 600-second budgets account for roughly 80% of the total time.

**Checkpointing:** the runner appends one row to `results/tables/full_experiments.csv` after every completed experiment and flushes the file. If the run is interrupted (Ctrl+C, power loss, sleep), restart the same command — completed experiments are skipped and the run resumes from the first incomplete tuple.

**Progress checks (in a second terminal):**

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("results/tables/full_experiments.csv")
with p.open() as f:
    rows = list(csv.DictReader(f))
c = [r for r in rows if r.get("status","").lower() == "completed"]
print(f"Completed: {len(c)}/{len(rows)}")
if c:
    last = c[-1]
    print(f"Last: {last['dataset']} | {last['method']} | "
          f"seed={last['seed']} | budget={last['budget_seconds']}s")
PY
```

Run this every 20–30 minutes in a second terminal. The `Completed` count should increase as the run progresses. If it hasn't changed in more than 12 minutes during a 600-second experiment, check CPU usage of the Python process to confirm it's still working.

**Trial-level progress:** each experiment prints `[START]` and `[END]` lines with elapsed time, so the terminal itself is a live progress log. Redirect stderr to a file to keep it clean:

```powershell
python experiments/run_full_experiments.py --final 2> warnings.log
```

**Narrowing the run for testing:** if you only want to verify a subset, temporarily edit `run_full_experiments.py`:

```python
SEEDS_FINAL = [42]
BUDGETS_FINAL = [30]
```

Then run `--final`. When you're ready to do the full matrix, restore the original values (`[42, 43, 44]` and `[30, 120, 600]`). Because the resume key includes dataset, method, seed, and budget, a narrowed run does not affect a later full run — the missing combinations will simply be added when you restore the matrix.

**Runtime variability:** absolute wall-clock times depend on your CPU and background load. The relative comparison between methods is what the experiment measures, not the absolute speed.

---

## 8. Regenerate the analysis

```bash
python experiments/analyze_final_results.py
```

Reads `results/tables/full_experiments.csv`, validates the 108-combination matrix, and writes:

- `results/tables/final_analysis.csv`
- `results/tables/final_analysis.md`

It also prints a terminal summary including:

- Matrix validation (expected vs found, duplicates, missing combinations)
- Dataset × Budget × Method F1 statistics (mean, std, median)
- Method × Budget aggregated statistics
- Budget effect (30 → 120 → 600 change in F1 and trial counts)
- Seed variability (min / mean / max / std per combination)
- Time-Aware Bayesian vs Standard Bayesian descriptive deltas
- Empirical patterns derived from the data

None of the output is a ranking or a superiority claim.

---

## 9. Regenerate the figures

```bash
python experiments/plot_final_results.py
```

Writes:

- `results/figures/best_f1_vs_budget.png`
- `results/figures/trials_vs_budget.png`
- `results/figures/runtime_vs_budget.png`
- `results/figures/bayesian_vs_timeaware.png`
- `results/figures/f1_seed_variability.png`
- `results/figures/time_to_target.png`
- `results/tables/final_figure_summary.md`

The `time_to_target.png` figure is only generated if the CSV contains populated `time_to_target` values. If those values are absent, the script prints a clear message and skips the figure.

---

## 10. Optional — launch the Streamlit dashboard

```bash
streamlit run src/timetuneml/app/streamlit_app.py
```

A browser tab opens at `http://localhost:8501`. The dashboard lets you:

- Pick one of the three datasets.
- Pick one of the four methods.
- Set a budget (10–60 s slider).
- Set a random seed.
- Click **Run Optimization** to execute the selected method against the selected dataset.

The dashboard is a thin UI layer over the same optimizers used by the experiment runner. It is not required for reproducing the 108-experiment matrix; it exists for exploratory use.

To stop the server, press `Ctrl+C` in the terminal where it's running.

---

## 11. Reproducibility guarantees

- **Same seed → same experiment.** Every experiment uses an explicit integer seed that controls the train/test split, the Random Forest's internal randomness, Random Search sampling, and Optuna TPE sampling.
- **Same split across methods.** For a given `(dataset, seed, budget)`, all four methods use the same train/test split. Preprocessing is fitted only on the training fold inside each trial.
- **No test-set leakage.** The test fold is used only for the final F1 evaluation of the configuration chosen by the search.
- **Deterministic matrix.** `--final --dry-run` prints the exact list of experiments that `--final` will run, in order.
- **Checkpointed persistence.** Every completed experiment is written to `full_experiments.csv` before the next one starts. Interrupting the run never loses completed work.

---

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'timetuneml'`**

The experiment scripts add `src/` to `sys.path` themselves. If you see this error, ensure you're running the script from the project root:

```bash
python experiments/run_full_experiments.py --final
```

**`sklearn.utils.parallel.delayed` warnings flooding the terminal**

These are cosmetic. The optimizers use `n_jobs=1` for the Random Forest, so the warning is emitted but harmless. To suppress it entirely:

- Redirect stderr to a file:
  ```bash
  python experiments/run_full_experiments.py --final 2> warnings.log
  ```
- Or set the environment variable before launching:
  - Windows PowerShell: `$env:PYTHONWARNINGS = "ignore::UserWarning"`
  - macOS/Linux: `PYTHONWARNINGS="ignore::UserWarning" python experiments/run_full_experiments.py --final`

**The runner appears to hang**

Check CPU usage of the Python process:

```powershell
Get-Process python | Select-Object Id, CPU, WorkingSet
```

Run the command twice, a few seconds apart:

- If `CPU` increases → the process is working. A 600-second budget means the CSV will not update for up to 10 minutes. Wait.
- If `CPU` is frozen for more than ~60 seconds → investigate. Press `Ctrl+C` once in the running terminal, wait for `Interrupted by user. Preserving completed results.`, then restart with the same command to resume from the last completed experiment.

**`full_experiments.csv` has mixed schemas from earlier milestones**

The current schema is defined in `FULL_COLUMNS` inside `run_full_experiments.py`. If you have a CSV from before the `dataset` column was added, older rows are preserved verbatim and new rows use the new schema. If this causes issues with the analysis script, back up the old file, delete it, and re-run `--final` from scratch:

```bash
mv results/tables/full_experiments.csv results/tables/full_experiments.csv.bak
python experiments/run_full_experiments.py --final
```

**Optuna prints its own startup message on every run**

Expected. Optuna emits a one-line banner the first time it's used in a process. It's informational, not an error.

**`streamlit` prompts for an email on first launch**

Streamlit shows a one-time onboarding prompt on first run per user. Press `Enter` with the email field blank to skip it. To suppress it entirely, create `~/.streamlit/credentials.toml` with:

```toml
[general]
email = ""
```

---

## 13. What the original run produced

For reference, the original 108-experiment matrix on a typical laptop produced:

- **108 / 108 experiments completed**
- **0 failures, 0 duplicates, 0 missing combinations**
- **Wall-clock time:** approximately 6 hours
- **Breast Cancer** search-method mean F1: **0.9704**
- **Iris** search-method mean F1: **0.9777**
- **Wine** search-method mean F1: **1.0000**
- **Largest reported cross-seed F1 standard deviation:** 0.0315
- **`time_to_target` populated in 72 of 108 rows**

Your numbers should be nearly identical if you use the same seeds and matrix, subject to small variation in wall-clock timings.

---

## 14. Files produced by a complete run

After running the steps above, the following files will exist:

**Tables (`results/tables/`):**

- `full_experiments.csv` — one row per experiment
- `experiment_summary.csv` — aggregated by (dataset, method, budget)
- `final_analysis.csv` — full descriptive statistics per (dataset, method, budget)
- `final_analysis.md` — Markdown version of the analysis
- `final_figure_summary.md` — index of generated figures

**Figures (`results/figures/`):**

- `best_f1_vs_budget.png`
- `trials_vs_budget.png`
- `runtime_vs_budget.png`
- `bayesian_vs_timeaware.png`
- `f1_seed_variability.png`
- `time_to_target.png` (conditional on data availability)

These are the artifacts referenced by `README.md` and `docs/FINAL_REPORT.md`.