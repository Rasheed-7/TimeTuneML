# TimeTuneML — Final Figure Summary

Generated from `results/tables/full_experiments.csv`. No experimental data was modified.

| Figure | What it shows | Dimensions represented |
| --- | --- | --- |
| `best_f1_vs_budget.png` | Mean Best F1 vs configured budget; one line per method. | dataset × method × budget |
| `trials_vs_budget.png` | Mean number of trials vs configured budget; one line per method. | dataset × method × budget |
| `runtime_vs_budget.png` | Mean actual runtime vs configured budget; one line per method. | dataset × method × budget |
| `bayesian_vs_timeaware.png` | Standard Bayesian vs Time-Aware Bayesian mean Best F1 by budget. | dataset × {Bayesian, Time-Aware Bayesian} × budget |
| `f1_seed_variability.png` | Mean ± std Best F1 across seeds, grouped by method and budget. | dataset × method × budget × seed |
| `time_to_target.png` | Mean time-to-target vs configured budget; one line per method. | dataset × method × budget |
