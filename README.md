 # ⏱️ TimeTuneML

 ### Time-Aware Hyperparameter Optimization Under Fixed Computational Budgets

 An experimental framework for studying whether computational cost can improve hyperparameter optimization when model training is constrained by a strict time budget.


 ## 📌 Overview

 Hyperparameter optimization (HPO) normally focuses on one question:

 > **Which configuration produces the best model?**

 But in real systems, computational resources are limited.

 A configuration that produces a slightly better model may take significantly longer to train than another configuration with almost the same performance.

 **TimeTuneML** explores this trade-off by treating both **model performance** and **execution cost** as observable signals during hyperparameter optimization.

 The project compares four strategies under controlled computational budgets:

 1. **Default Random Forest** — single baseline model
2. **Random Search**
3. **Bayesian Optimization using Optuna/TPE**
4. **Time-Aware Bayesian Optimization**

 The goal is not to claim a new Bayesian optimization algorithm, but to build a reproducible experimental framework for measuring how optimization strategies behave when computation is explicitly budgeted.

---

 ## 🎯 Research Question

 > **Can incorporating computational cost into Bayesian hyperparameter optimization improve the quality of models discovered within a fixed time budget?**

 This project treats the question as an **empirical experiment** rather than assuming the answer beforehand.

 The final evaluation consists of:

 **3 datasets × 4 methods × 3 seeds × 3 budgets = 108 experiments**

---

 ## 🧠 Core Idea

 Traditional hyperparameter optimization can be viewed as:

```
Find configuration x that maximizes model performance f(x)
```

 TimeTuneML additionally records the computational cost:

```
Performance → f(x)
Execution Cost → g(x)
```

 while enforcing a global budget:

```
Σ trial_time ≤ B
```

 This makes computational cost a first-class experimental signal.

 ### Time-Aware Optimization

 The Time-Aware Bayesian workflow uses a lightweight execution-cost estimator.

 During an initial warm-up phase, it behaves like standard Bayesian optimization.

 After warm-up:

```
Propose candidate
      ↓
Estimate execution cost
      ↓
Check remaining budget
      ↓
Candidate likely fits?
   ↙          ↘
 Yes           No
 ↓              ↓
Evaluate     Consider another
 ↓
Record actual cost
 ↓
Update search
```

 The actual measured runtime remains authoritative. Predicted cost is only used to avoid obviously expensive candidates that are unlikely to fit within the remaining budget.

---

 # 🏗️ Architecture

```
                    ┌─────────────────┐
                    │     Dataset     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Validation   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Preprocess   │
                    │  Leakage-Safe   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Model Registry │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Search Space  │
                    └────────┬────────┘
                             ↓
              ┌──────────────────────────────┐
              │      Optimization Engine    │
              │                              │
              │  • Default RF                │
              │  • Random Search             │
              │  • Bayesian / TPE            │
              │  • Time-Aware Bayesian       │
              └──────────────┬───────────────┘
                             ↓
                    ┌─────────────────┐
                    │  Trial Manager  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Budget Control  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Final Evaluation│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Experiment Logs │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │Analysis/Figures │
                    └─────────────────┘
```

---

 ## 🔬 Optimization Strategies

 | Strategy | Description |
| --- | --- |
| **Default RF** | A single Random Forest configuration used as a baseline |
| **Random Search** | Randomly samples configurations from the predefined search space |
| **Bayesian Optimization** | Uses Optuna's TPE sampler to propose promising configurations |
| **Time-Aware Bayesian** | Adds execution-cost estimation and remaining-budget awareness to the Bayesian workflow |

### Default Random Forest

 The baseline trains one Random Forest model per experiment.

 The configured search budget is recorded for comparison, but the baseline does not consume the search budget.

 ### Random Search

 Random Search samples hyperparameter configurations from the predefined search space and continues until the global time budget is exhausted.

 ### Bayesian Optimization

 TimeTuneML uses **Optuna's Tree-structured Parzen Estimator (TPE)** to sequentially propose hyperparameter configurations based on previous trial observations.

 ### Time-Aware Bayesian Optimization

 The project-specific time-aware variant extends the Bayesian workflow with:

 - measured trial duration
- cumulative elapsed time
- remaining budget
- lightweight execution-cost estimation
- cost-aware candidate filtering
- recorded performance/cost utility

 > **Important:** TimeTuneML does not claim to introduce a novel Bayesian optimization algorithm. It is an experimental framework inspired by established cost-aware optimization ideas.

---

 ## ⏱️ Computational Budget

 Every search-based experiment operates under a fixed global time budget.

 Each trial records:

```
Parameters
Score
Duration
Cumulative Time
Status
Optimizer
Dataset
Model
Seed
```

 The budget controller:

 - tracks elapsed wall-clock time
- measures every trial using `time.perf_counter()`
- checks the remaining budget before starting another trial
- stops the search when the budget is exhausted
- records actual execution time
- handles failed trials without corrupting the experiment

---

 ## 🔒 Leakage-Safe Evaluation

 TimeTuneML follows a strict train/test separation.

```
Raw Dataset
     ↓
Train / Test Split
     ↓
Training Data
     ↓
Preprocessing + Cross Validation
     ↓
Hyperparameter Optimization
     ↓
Best Configuration
     ↓
Final Model
     ↓
Test Set Evaluation
```

 The test set is **not used during hyperparameter search or model selection**.

 Preprocessing is fitted on the training data and then applied to the test data.

 This keeps the final evaluation isolated from the optimization process.

---

 ## 🧪 Experimental Setup

 | Property | Configuration |
| --- | --- |
| Datasets | Breast Cancer, Iris, Wine |
| Model | `RandomForestClassifier` |
| Preprocessing | Median imputation + standard scaling |
| Metric | F1 |
| Seeds | 42, 43, 44 |
| Search Budgets | 30s, 120s, 600s |
| Optimization Methods | 4 |
| Total Experiments | **108** |

### Experiment Matrix

```
3 datasets
    ×
4 optimization strategies
    ×
3 random seeds
    ×
3 time budgets
    =
108 experiments
```

---

 # 📊 Results

 All **108 experiments completed successfully**:

 - 108 expected
- 108 completed
- 0 failed
- 0 duplicate experiment combinations
- 0 missing combinations

 ### Aggregate Mean F1

 | Dataset | Search Methods Mean F1 |
| --- | --- |
| Breast Cancer | **0.9704** |
| Iris | **0.9777** |
| Wine | **1.0000** |

Across the reported budgets, the search-method mean F1 values remained unchanged for these three datasets.

 This is important because the datasets are small and relatively inexpensive to train on, which limits the amount of performance headroom available to the optimization strategies.

 ### Bayesian vs Time-Aware Bayesian

 In the final aggregate results:

 > **Time-Aware Bayesian Optimization and Standard Bayesian Optimization had equal mean F1 in every dataset/budget combination.**

 However, their:

 - trial counts
- runtimes
- candidate-selection behavior

 differed in some experimental conditions.

 Therefore, this project does **not** claim that the Time-Aware method improved F1.

 The experiments instead demonstrate how the framework can measure performance and computational behavior under controlled budgets.

---

 ## 📈 Visual Results

 The final experiment generated six figures.

 | Figure | Purpose |
| --- | --- |
| `best_f1_vs_budget.png` | Mean Best F1 vs configured budget |
| `trials_vs_budget.png` | Mean number of trials vs budget |
| `runtime_vs_budget.png` | Mean runtime vs budget |
| `bayesian_vs_timeaware.png` | Bayesian vs Time-Aware Bayesian comparison |
| `f1_seed_variability.png` | Mean ± standard deviation across seeds |
| `time_to_target.png` | Time required to reach the target threshold |

All figures are available under:

```
results/figures/
```

---

 ## 🖥️ Streamlit Dashboard

 TimeTuneML includes a Streamlit interface for running and inspecting optimization experiments.

 The dashboard allows users to configure:

 - Dataset
- Model
- Optimization strategy
- Metric
- Time budget
- Random seed

 and view:

 - best configuration
- best score
- trial history
- convergence
- runtime information
- time-aware optimization information

 ### Run the dashboard

```
streamlit run src/timetuneml/app/streamlit_app.py
```

---

 ## ⚡ Quick Start

 ### 1\. Clone the repository

```
git clone https://github.com/Rasheed-7/TimeTuneML.git
cd TimeTuneML
```

 ### 2\. Create a virtual environment

 #### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

 #### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
```

 ### 3\. Install dependencies

```
pip install -r requirements.txt
```

 ### 4\. Run the tests

```
pytest
```

 ### 5\. Run a baseline experiment

```
python experiments/run_baseline.py
```

 ### 6\. Run the full experiment matrix

```
python experiments/run_full_experiments.py --final
```

 > The final matrix contains 108 experiments and can take significant time to complete.

 The runner checkpoints after each completed experiment and can resume from existing results.

 ### 7\. Regenerate figures

```
python experiments/plot_final_results.py
```

 ### 8\. Launch the dashboard

```
streamlit run src/timetuneml/app/streamlit_app.py
```

---

 ## 📁 Project Structure

```
TimeTuneML/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── FINAL_REPORT.md
│   └── REPRODUCIBILITY.md
│
├── src/
│   └── timetuneml/
│       │
│       ├── app/
│       │   └── streamlit_app.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── validation.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── metrics.py
│       │
│       ├── models/
│       │   ├── registry.py
│       │   └── search_spaces.py
│       │
│       ├── optimization/
│       │   ├── base.py
│       │   ├── bayesian.py
│       │   ├── budget.py
│       │   ├── cost_aware.py
│       │   └── random_search.py
│       │
│       └── preprocessing/
│           └── pipeline.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_full_experiments.py
│   ├── analyze_final_results.py
│   └── plot_final_results.py
│
├── results/
│   ├── figures/
│   └── tables/
│
└── tests/
    ├── test_data.py
    ├── test_search_space.py
    └── test_budget.py
```

---

 ## 🛠️ Tech Stack

 ### Machine Learning

 - Python
- Scikit-learn
- Random Forest
- Cross-validation
- F1 evaluation

 ### Optimization

 - Optuna
- Tree-structured Parzen Estimator (TPE)
- Random Search
- Time-aware candidate selection
- Computational budget control

 ### Data & Analysis

 - NumPy
- Pandas
- Matplotlib
- CSV-based experiment tracking

 ### Application

 - Streamlit

 ### Engineering

 - Modular Python architecture
- Reproducible experiments
- Automated tests
- Seed-controlled experimentation
- Checkpointed experiment execution

---

 ## 🔁 Reproducibility

 Experiments use explicit random seeds.

 The final experimental matrix uses:

```
Seeds:
42
43
44

Budgets:
30 seconds
120 seconds
600 seconds
```

 The experiment runner checkpoints after each completed experiment.

 This means an interrupted experiment run can be resumed without unnecessarily repeating completed combinations.

 ### Result Files

 | File | Purpose |
| --- | --- |
| `full_experiments.csv` | Raw experiment-level results |
| `experiment_summary.csv` | Aggregated experiment results |
| `final_analysis.csv` | Final descriptive statistics |
| `final_analysis.md` | Markdown analysis |
| `final_figure_summary.md` | Figure index |

---

 ## 🧪 Testing

 TimeTuneML includes a lightweight automated test suite covering:

 - dataset loading
- search-space validity
- computational budget behavior

 Run:

```
pytest
```

 The tests are intentionally small and fast so they can be used as a sanity check before running experiments.

---

 ## ⚠️ Limitations

 The results should be interpreted within the scope of the experiment.

 ### 1\. Small datasets

 Only three classical tabular datasets were evaluated.

 They are relatively small and inexpensive to train on.

 ### 2\. Single model family

 The final experiment matrix uses `RandomForestClassifier`.

 Results should not automatically be generalized to other model families.

 ### 3\. Limited seeds

 Three seeds provide some visibility into variability, but they are not sufficient for highly precise statistical conclusions.

 ### 4\. Hardware dependence

 Execution time depends on the machine and environment where the experiments are executed.

 ### 5\. Noisy trial cost

 The same configuration can have slightly different wall-clock execution times across runs.

 ### 6\. Search-space dependence

 Changing the hyperparameter search space can change the behavior and relative performance of the optimization strategies.

 ### 7\. Limited performance headroom

 Some datasets reach high F1 scores quickly, making it difficult for any optimizer to demonstrate substantial improvement.

 ### 8\. No universal superiority

 These experiments do not establish that one optimization strategy is universally better than another.

---

 ## 🔮 Future Work

 Possible extensions include:

 - larger and more computationally heterogeneous datasets
- more expensive model families
- additional random seeds
- confidence intervals and paired statistical comparisons
- stronger cost-aware acquisition strategies
- more sophisticated execution-cost models
- experiments where trial costs vary substantially across configurations

 These are intentionally outside the scope of the current completed experiment.

---

 ## 📚 Documentation

 Additional documentation:

 - `FINAL_REPORT.md` — detailed research report
- `REPRODUCIBILITY.md` — reproducibility information
- `results/tables/` — experiment results and analysis tables
- `results/figures/` — generated visualizations

---

 ## 💡 Key Takeaway

 TimeTuneML demonstrates a complete experimental workflow for studying hyperparameter optimization under computational constraints.

 The project combines:

```
Hyperparameter Optimization
          +
Execution-Cost Measurement
          +
Global Time Budgets
          +
Reproducible Experiments
          +
Quantitative Analysis
```

 The final 108-experiment study provides an empirical comparison of Random Search, Bayesian Optimization, and a project-specific Time-Aware Bayesian workflow across multiple datasets, seeds, and computational budgets.

 The experiments **do not establish a universal advantage for cost-aware optimization**. Instead, they provide a reproducible framework and measured evidence showing how optimization behavior changes under fixed computational budgets.

---

 ## 👤 Project Status

 **Status: Completed Experimental Project**

 The core implementation, experimental evaluation, analysis, visualizations, dashboard, and documentation are complete.

---

 ### ⏱️ TimeTuneML

 **Measure performance. Measure cost. Optimize under constraints.**