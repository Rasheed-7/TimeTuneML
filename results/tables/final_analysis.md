# TimeTuneML — Final Analysis

Descriptive statistics only. No ranking or superiority claim is made.

## Matrix Validation

- Expected combinations : 108
- CSV rows              : 108
- Completed             : 108
- Failed                : 0
- Duplicate keys        : 0
- Missing combinations  : 0

## Results by Dataset, Budget, and Method

| Dataset | Method | Budget (s) | n | mean F1 | std F1 | median F1 | mean trials | mean runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Breast Cancer | Bayesian Optimization | 30 | 3 | 0.9704 | 0.0114 | 0.9655 | 42.3 | 30.35 |
| Breast Cancer | Default RF | 30 | 3 | 0.9704 | 0.0114 | 0.9655 | 1.0 | 0.27 |
| Breast Cancer | Random Search | 30 | 3 | 0.9704 | 0.0114 | 0.9655 | 79.3 | 30.54 |
| Breast Cancer | Time-Aware Bayesian Optimization | 30 | 3 | 0.9704 | 0.0114 | 0.9655 | 44.0 | 30.58 |
| Breast Cancer | Bayesian Optimization | 120 | 3 | 0.9704 | 0.0114 | 0.9655 | 204.7 | 120.32 |
| Breast Cancer | Default RF | 120 | 3 | 0.9704 | 0.0114 | 0.9655 | 1.0 | 0.35 |
| Breast Cancer | Random Search | 120 | 3 | 0.9704 | 0.0114 | 0.9655 | 273.7 | 120.49 |
| Breast Cancer | Time-Aware Bayesian Optimization | 120 | 3 | 0.9704 | 0.0114 | 0.9655 | 218.7 | 120.32 |
| Breast Cancer | Bayesian Optimization | 600 | 3 | 0.9704 | 0.0114 | 0.9655 | 887.3 | 467.70 |
| Breast Cancer | Default RF | 600 | 3 | 0.9704 | 0.0114 | 0.9655 | 1.0 | 0.30 |
| Breast Cancer | Random Search | 600 | 3 | 0.9704 | 0.0114 | 0.9655 | 1000.0 | 372.99 |
| Breast Cancer | Time-Aware Bayesian Optimization | 600 | 3 | 0.9704 | 0.0114 | 0.9655 | 1000.0 | 469.62 |
| Iris | Bayesian Optimization | 30 | 3 | 0.9777 | 0.0158 | 0.9666 | 109.0 | 30.24 |
| Iris | Default RF | 30 | 3 | 0.9443 | 0.0315 | 0.9666 | 1.0 | 0.26 |
| Iris | Random Search | 30 | 3 | 0.9777 | 0.0158 | 0.9666 | 81.7 | 30.28 |
| Iris | Time-Aware Bayesian Optimization | 30 | 3 | 0.9777 | 0.0158 | 0.9666 | 107.7 | 30.42 |
| Iris | Bayesian Optimization | 120 | 3 | 0.9777 | 0.0158 | 0.9666 | 661.3 | 116.73 |
| Iris | Default RF | 120 | 3 | 0.9443 | 0.0315 | 0.9666 | 1.0 | 0.14 |
| Iris | Random Search | 120 | 3 | 0.9777 | 0.0158 | 0.9666 | 427.0 | 120.21 |
| Iris | Time-Aware Bayesian Optimization | 120 | 3 | 0.9777 | 0.0158 | 0.9666 | 665.3 | 110.34 |
| Iris | Bayesian Optimization | 600 | 3 | 0.9777 | 0.0158 | 0.9666 | 1000.0 | 208.07 |
| Iris | Default RF | 600 | 3 | 0.9443 | 0.0315 | 0.9666 | 1.0 | 0.14 |
| Iris | Random Search | 600 | 3 | 0.9777 | 0.0158 | 0.9666 | 1000.0 | 281.44 |
| Iris | Time-Aware Bayesian Optimization | 600 | 3 | 0.9777 | 0.0158 | 0.9666 | 1000.0 | 209.45 |
| Wine | Bayesian Optimization | 30 | 3 | 1.0000 | 0.0000 | 1.0000 | 86.3 | 30.18 |
| Wine | Default RF | 30 | 3 | 0.9906 | 0.0133 | 1.0000 | 1.0 | 0.32 |
| Wine | Random Search | 30 | 3 | 1.0000 | 0.0000 | 1.0000 | 42.3 | 30.60 |
| Wine | Time-Aware Bayesian Optimization | 30 | 3 | 1.0000 | 0.0000 | 1.0000 | 78.3 | 30.27 |
| Wine | Bayesian Optimization | 120 | 3 | 1.0000 | 0.0000 | 1.0000 | 633.7 | 120.16 |
| Wine | Default RF | 120 | 3 | 0.9906 | 0.0133 | 1.0000 | 1.0 | 0.25 |
| Wine | Random Search | 120 | 3 | 1.0000 | 0.0000 | 1.0000 | 177.7 | 120.17 |
| Wine | Time-Aware Bayesian Optimization | 120 | 3 | 1.0000 | 0.0000 | 1.0000 | 618.3 | 120.17 |
| Wine | Bayesian Optimization | 600 | 3 | 1.0000 | 0.0000 | 1.0000 | 1000.0 | 226.06 |
| Wine | Default RF | 600 | 3 | 0.9906 | 0.0133 | 1.0000 | 1.0 | 0.26 |
| Wine | Random Search | 600 | 3 | 1.0000 | 0.0000 | 1.0000 | 911.3 | 600.25 |
| Wine | Time-Aware Bayesian Optimization | 600 | 3 | 1.0000 | 0.0000 | 1.0000 | 1000.0 | 207.09 |

## Time-Aware Bayesian vs Standard Bayesian (descriptive)

| Dataset | Budget (s) | Δ mean F1 | Δ mean runtime (s) | Δ mean trials |
| --- | ---: | ---: | ---: | ---: |
| Breast Cancer | 30 | 0.0000 | 0.23 | 1.7 |
| Breast Cancer | 120 | 0.0000 | -0.00 | 14.0 |
| Breast Cancer | 600 | 0.0000 | 1.92 | 112.7 |
| Iris | 30 | 0.0000 | 0.18 | -1.3 |
| Iris | 120 | 0.0000 | -6.39 | 4.0 |
| Iris | 600 | 0.0000 | 1.38 | 0.0 |
| Wine | 30 | 0.0000 | 0.09 | -8.0 |
| Wine | 120 | 0.0000 | 0.01 | -15.3 |
| Wine | 600 | 0.0000 | -18.97 | 0.0 |

## Time-to-Target

72 of 108 rows carry a `time_to_target` value.

## Empirical Patterns

- On at least one dataset/method, mean F1 at 600s is within 0.005 of mean F1 at 30s (plateau behaviour).
- For all search methods, mean trial count at 600s exceeds mean trial count at 30s.
- Time-Aware Bayesian and standard Bayesian differ by more than 0.005 mean F1 or more than 5 mean trials on at least one dataset/budget.
- Largest observed F1 standard deviation across seeds is 0.0315.
- Breast Cancer: search-method mean F1 ranges from 0.9704 to 0.9704 across all budgets.
- Iris: search-method mean F1 ranges from 0.9777 to 0.9777 across all budgets.
- Wine: search-method mean F1 ranges from 1.0000 to 1.0000 across all budgets.
