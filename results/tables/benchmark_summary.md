# TimeTuneML Benchmark Summary

## Experimental Setup

- **Dataset:** Breast Cancer (sklearn `load_breast_cancer`)
- **Model:** Random Forest Classifier
- **Metric:** F1
- **Seed:** 42
- **Budget:** 30 seconds

## Results

| Method | Best F1 | Total Time (s) | Number of Trials |
| --- | ---: | ---: | ---: |
| Default RF | 0.9655 | 0.4680 | 1 |
| Random Search | 0.9655 | 30.5119 | 60 |
| Bayesian Optimization | 0.9655 | 30.0470 | 55 |
| Time-Aware Bayesian | 0.9655 | 30.6496 | 52 |

## Observations

- All four methods reached the same best F1 value on this single 30-second, seed-42 run.
- The Default RF completed in substantially less time because it performed a single training run and no search.
- The three search-based methods each consumed approximately the configured 30-second budget.
- Within that budget the search methods evaluated different numbers of trials, reflecting differences in sampled hyperparameters and per-trial runtimes.
- These results describe one experiment only and should not be used to rank the methods; multi-seed and multi-budget runs are required before drawing general conclusions.
