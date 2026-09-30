# TimeTuneML — Final Report

**Time-aware hyperparameter optimization under fixed computational budgets**

---

## 1. Abstract

We present TimeTuneML, an experimental framework for comparing hyperparameter optimization strategies under a fixed global time budget. The framework implements four methods — a Default Random Forest baseline, Random Search, Optuna/TPE-based Bayesian Optimization, and a project-specific Time-Aware Bayesian variant that incorporates measured execution cost into trial selection — and evaluates them across three built-in scikit-learn classification datasets (Breast Cancer, Iris, Wine), three seeds, and three budgets (30 s, 120 s, 600 s). All 108 experiments completed successfully. Across the reported aggregate table, the four methods reached similar best F1 values, and Time-Aware Bayesian Optimization did not achieve a higher mean F1 than standard Bayesian Optimization. Trial counts scaled with budget, and seed-to-seed variability was small but non-zero. The study does not establish a general superiority claim for any method; the results describe one experimental configuration on three small tabular datasets.

---

## 2. Introduction

Hyperparameter optimization (HPO) is a routine but computationally expensive step in modern machine learning. When trials are cheap, the search algorithm's sample efficiency is the dominant concern. When trials are expensive — large models, large datasets, long training times — the *cost* of each trial becomes a first-class part of the optimization problem. TimeTuneML was built to make that cost signal explicit, measurable, and comparable across search strategies.

The project does not introduce a new Bayesian optimization algorithm. It is a small research harness that reuses established components (Optuna's TPE sampler for Bayesian search, scikit-learn's Random Forest as the model family) and adds a project-specific cost-aware layer on top.

---

## 3. Problem Definition

Consider the following constrained optimization problem:

> Given a fixed global time budget `B`, a search space `H`, and an objective `f : H → ℝ` that is expensive to evaluate, choose a sequence of configurations `h₁, h₂, …` such that the best `f(hᵢ)` observed after the total elapsed time reaches `B` is as high as possible.

Each evaluation of `f` produces two quantities:

- a **performance** value (here: test F1 of the trained model),
- a **cost** value (here: wall-clock seconds spent training and evaluating).

A cost-blind optimizer maximizes the performance signal alone. A cost-aware optimizer considers both.

---

## 4. Research Question

> **Can incorporating computational cost into Bayesian hyperparameter optimization improve the quality of models discovered within a fixed time budget?**

This is posed as an empirical question. The project does not assume an answer. The experimental design is intended to produce observations that either support or fail to support the hypothesis; both outcomes are informative.

---

## 5. Methodology

The framework proceeds in the following steps for every experiment:

1. Load the dataset and validate its structure.
2. Create a stratified train/test split with a fixed seed.
3. Run the selected optimizer for up to `B` seconds. Each trial:
   - samples a configuration from the search space,
   - fits preprocessing on the training fold only,
   - fits the model on the transformed training fold,
   - evaluates on the held-out test fold,
   - records performance, duration, and cumulative elapsed time.
4. Record the best configuration and its metrics.
5. Log the experiment row to a CSV.

The test set is never inspected during the search; it is only used at the end of each trial to compute the reported F1. Preprocessing statistics are learned exclusively from the training fold.

---

## 6. System Architecture

The framework is organized into discrete modules so that each piece can be tested independently.

| Layer | Responsibility |
|---|---|
| Data | Load and validate sklearn datasets |
| Preprocessing | Column-wise imputation and scaling, fitted on training data |
| Model | Registry of the baseline Random Forest and model factory |
| Search Space | Declarative hyperparameter ranges |
| Optimization | Random Search, Bayesian (TPE), Time-Aware Bayesian |
| Trial | Per-trial record: parameters, score, duration, cumulative time, status |
| Budget | Monotonic global timer with `start`, `elapsed`, `remaining`, `is_exhausted` |
| Evaluation | F1, accuracy, precision, recall, ROC-AUC |
| Tracking | Append-and-flush CSV for crash-safe persistence |
| Analysis | Descriptive statistics and generated figures |

---

## 7. Optimization Strategies

### Default Random Forest

A single model trained with default hyperparameters. Serves as a reference point rather than a search method.

### Random Search

Samples configurations uniformly at random from the search space until the budget is exhausted. No use of past trial information.

### Bayesian Optimization

Uses Optuna's Tree-structured Parzen Estimator. Each completed trial updates a probabilistic surrogate; the next candidate is drawn to maximize expected improvement under that surrogate. Also stops when the budget is exhausted.

### Time-Aware Bayesian Optimization

Extends the Bayesian workflow with a cost-aware layer:

- **Warm-up.** For the first `warmup_trials` trials, behaves identically to standard Bayesian optimization, collecting `(configuration, measured duration)` pairs.
- **Cost model.** A lightweight Ridge regression predicts trial duration from hyperparameters. The model is refitted as observations accumulate; it never sees the F1 signal.
- **Rejection rule.** After warm-up, if the predicted duration of a proposed candidate exceeds the remaining budget, the candidate is rejected and a fresh one is sampled.
- **Utility.** A scalar `performance / max(predicted_cost, ε)` is recorded per trial for later analysis.

This is a **project-specific experimental framework** inspired by established cost-aware optimization ideas. It does not introduce a new Bayesian algorithm, and the project does not claim otherwise.

---

## 8. Computational Budgeting

A single `BudgetController` object manages the global budget:

```python
elapsed_time = time.perf_counter() - start_time
remaining_time = max(0.0, budget_seconds - elapsed_time)
budget_exhausted = elapsed_time >= budget_seconds
```

Two important properties:

1. **Monotonic clock.** `time.perf_counter()` is used, not wall-clock time, so NTP corrections and system-clock changes do not affect the budget.
2. **Authoritative measurement.** The actual measured duration is what counts. The cost model is only used to *avoid* obviously too-expensive candidates; it never replaces a real measurement.

The loop checks the budget *before* starting each new trial. If the budget is exhausted, the loop exits cleanly and the last completed trial is preserved.

---

## 9. Dataset and Model Setup

| Property | Value |
|---|---|
| Datasets | `load_breast_cancer`, `load_iris`, `load_wine` |
| Model | `sklearn.ensemble.RandomForestClassifier` |
| Preprocessing | `SimpleImputer` + `StandardScaler` (numeric path) |
| Metric | F1 |
| Search space | `n_estimators` ∈ [50, 500], `max_depth` ∈ [3, 30], `min_samples_split` ∈ [2, 20], `min_samples_leaf` ∈ [1, 10], `max_features` ∈ {sqrt, log2, 0.3, 0.5, 1.0} |

All preprocessing is fitted inside the optimizer loop on the training fold of the current experiment.

---

## 10. Experimental Design

### Matrix

| Dimension | Values |
|---|---|
| Datasets | Breast Cancer, Iris, Wine |
| Methods | Default RF, Random Search, Bayesian Optimization, Time-Aware Bayesian Optimization |
| Seeds | 42, 43, 44 |
| Budgets | 30 s, 120 s, 600 s |

**Total: 3 × 4 × 3 × 3 = 108 experiments.**

### Fairness Controls

- Same train/test split configuration across methods (same seed, same `test_size`, same stratify).
- Same preprocessing pipeline for all methods.
- Same model family for all methods.
- Same search space for all search methods.
- Same metric (F1) for optimization and evaluation.
- Same Random Forest `random_state` (per-experiment seed).

### Persistence

Each experiment's result is appended to `results/tables/full_experiments.csv` and flushed before the next experiment begins. This makes the run resumable after any interruption.

---

## 11. Evaluation Metrics

Per trial, the following metrics are recorded on the held-out test fold:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

F1 is the primary optimization metric. The other metrics are recorded for diagnostic purposes.

---

## 12. Results

All 108 experiments completed. No failures, no duplicates, no missing combinations.

### Aggregate Observations

- **Breast Cancer** search-method mean F1: **0.9704** across the reported budgets.
- **Iris** search-method mean F1: **0.9777** across the reported budgets.
- **Wine** search-method mean F1: **1.0000** across the reported budgets.
- Trial counts increased substantially with larger budgets for all search methods.
- Largest reported cross-seed F1 standard deviation: **0.0315**.
- **72 of 108 rows** carried a populated `time_to_target` value.

These numbers are drawn directly from `results/tables/final_analysis.csv`. No interpretative language (better/worse/winner) is applied to them.

### Figures

- `results/figures/best_f1_vs_budget.png`
- `results/figures/trials_vs_budget.png`
- `results/figures/runtime_vs_budget.png`
- `results/figures/bayesian_vs_timeaware.png`
- `results/figures/f1_seed_variability.png`
- `results/figures/time_to_target.png`

---

## 13. Time-Aware vs Standard Bayesian Analysis

The following is a **descriptive comparison**, not a statistical test.

| Dimension | Observation |
|---|---|
| Mean F1 | Equal in every dataset / budget combination in the reported aggregate table |
| Mean runtime | Differed in some conditions |
| Mean trial count | Differed in some conditions |
| Best utility | Recorded for Time-Aware runs only; not directly comparable to a Bayesian counterpart |

**Interpretation guidance:** equality of mean F1 in the aggregate table is consistent with several explanations (the search space may be easy enough that both methods find the ceiling quickly; the budgets may be short enough that neither method has exhausted its exploration; the test set may be too small to distinguish small effects). The data do not, and are not intended to, decide between those explanations.

---

## 14. Time-to-Target Analysis

The CSV includes a `time_to_target` column, populated in 72 of 108 rows. Where present, it records the earliest cumulative time at which best-so-far F1 reached the target derived from the best observed F1 for the corresponding dataset.

The figure `results/figures/time_to_target.png` visualizes mean time-to-target per method, dataset, and budget where values were available.

**What is not claimed:** no conclusion is drawn about which method reaches the target "fastest" in a general sense. The metric is descriptive and is only meaningful for the specific target values and the specific datasets on which it was computed.

---

## 15. Limitations

- **Three small tabular datasets.** None of them stresses the optimizers.
- **Random Forest only.** The findings do not automatically transfer to other model families.
- **Three seeds.** Sufficient to observe variability, not sufficient for tight statistical inference.
- **Hardware- and environment-dependent timing.** Absolute cost values only have meaning on the machine the experiments were run on.
- **Trial cost is noisy.** Even for the same configuration, wall-clock time varies between runs.
- **Search-space design affects conclusions.** A wider or narrower space would change how much headroom each method has.
- **Some datasets reach high F1 quickly.** This compresses the amount of observable headroom for any strategy.
- **No universal superiority claim.** The results do not show that any method is "better" in a general sense.
- **Cost-aware behavior may matter more elsewhere.** Larger, more heterogeneous workloads could produce a different signal — but this is not what was measured.

---

## 16. Reproducibility

### Environment

- Python 3.10+
- `numpy`, `scikit-learn`, `matplotlib`, `optuna`, `streamlit`

### Running the experiment

```bash
python experiments/run_full_experiments.py --final
```

The script is resumable: on restart, experiments whose `(dataset, method, seed, budget)` tuple already appears in `results/tables/full_experiments.csv` with `status="completed"` are skipped.

### Regenerating the analysis

```bash
python experiments/analyze_final_results.py
```

### Regenerating the figures

```bash
python experiments/plot_final_results.py
```

### Seeds

Every experiment uses an explicit integer seed. Same seed → same split → same Random Forest initialization → same sequence of proposed hyperparameters (up to the underlying libraries' own determinism).

### Result files

See `results/tables/` for all CSVs and Markdown reports. See `results/figures/` for all PNGs.

---

## 17. Conclusion

TimeTuneML is a working experimental framework for comparing hyperparameter optimization strategies under a fixed computational budget. Across 108 experiments on three small tabular datasets:

- All four methods completed successfully.
- Best F1 was similar across methods, with some datasets reaching ceiling values quickly.
- Trial counts scaled with the configured budget.
- Time-Aware Bayesian Optimization and standard Bayesian Optimization had equal mean F1 in every reported aggregate dataset/budget cell.
- Seed variability was small but non-zero.

**The research question is not definitively answered.** The observed results do not support the claim that cost-aware Bayesian optimization improves F1 in this setting, and they also do not refute it in any general sense. Whether the cost-aware layer provides measurable benefit under different conditions — larger datasets, more expensive models, wider search spaces — is left open and would require a different experimental configuration to investigate.

---

## 18. Future Work

- Larger and more heterogeneous datasets.
- More expensive model families where per-trial cost varies substantially.
- More seeds for tighter statistics.
- Stronger cost-aware acquisition strategies that use the recorded utility signal.
- Additional statistical analysis (paired comparisons, effect sizes) once the sample size supports it.

These are open directions, not commitments.