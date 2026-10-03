# BenchForge

BenchForge is reproducible benchmarking infrastructure for classical machine learning. It turns a
strict declarative configuration into leakage-safe evidence that can be inspected and reproduced.

BenchForge V0.3 has three workflow levels:

- `benchforge run`: execute one concrete pipeline.
- `benchforge benchmark`: compare several concrete pipelines on shared folds.
- `benchforge search`: tune several model families under a finite budget and compare the tuning
  procedures using untouched, shared outer folds.

Future AutoML may orchestrate the search layer, but automatic pipeline invention is not V0.3.

## Installation and examples

BenchForge requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

benchforge run configs/examples/breast_cancer_logreg.yaml
benchforge benchmark configs/examples/breast_cancer_suite.yaml
benchforge benchmark configs/examples/mixed_tabular_suite.yaml
benchforge search configs/examples/breast_cancer_search.yaml
benchforge search configs/examples/mixed_tabular_search.yaml
```

The Python API exposes `load_run_config()` / `run_benchmark()`,
`load_benchmark_config()` / `run_benchmark_suite()`, and
`load_search_config()` / `run_search()`.

## Search configuration

Search uses its own strict `SearchConfig` instead of expanding `BenchmarkConfig`:

```yaml
schema_version: 1
task: binary_classification
dataset:
  name: breast_cancer
outer_split: {strategy: stratified_kfold, n_splits: 5, shuffle: true}
inner_split: {strategy: stratified_kfold, n_splits: 3, shuffle: true}
preprocessing: {median_imputation: true, standardize: true}
models:
  - {name: logistic_regression, id: logistic, mode: search}
  - name: random_forest_classifier
    id: forest
    mode: search
    parameters: {n_estimators: 200}
  - {name: dummy_classifier, id: dummy, mode: fixed}
metrics: [roc_auc, balanced_accuracy, f1]
primary_metric: roc_auc
budget:
  trials_per_outer_fold: 20
  # timeout_seconds_per_outer_fold: 600
final_search: {enabled: true, trials: 40}
seed: 42
output: {directory: artifacts}
```

Every outer training partition receives its own inner Optuna search. BenchForge does not tune on
the same held-out fold it uses to evaluate a tuned model. Inner preprocessing and selection see
only outer-training rows; the outer validation partition stays untouched until parameters have
been selected. All families share the same outer folds. The tuned leaderboard is ranked only by
aggregate outer-fold primary-metric means, never by Optuna scores.

`mode: fixed` evaluates a reference configuration directly on the shared outer folds without an
inner-search cost. Candidate `parameters` are fixed overrides. A fixed parameter is removed from
the curated search space; the search space is never allowed to overwrite it.

The primary deterministic budget is `trials_per_outer_fold`. An optional timeout may stop a study
earlier; when both are present, the first reached limit stops the study. Timeout is not an exact
runtime budget. Search is sequential (`n_jobs=1`) and uses deterministic TPE, inner-fold, and
estimator seeds derived with SHA-256.

Nested evaluation is intentionally more expensive. A rough fit count is:

```text
outer folds × trials per outer fold × inner folds × searchable families
```

plus one outer refit per family/fold, fixed-baseline fits, and an optional final search. This is a
compute explanation, not a wall-clock promise.

After outer ranking, `final_search` can tune the top measured family on all available data. Its CV
score is selection evidence for deployable hyperparameters, not unbiased held-out performance. It
cannot change the already-computed nested leaderboard.

## Data and leakage-safe preprocessing

BenchForge includes the sklearn Breast Cancer Wisconsin dataset and reads local CSV or Parquet:

```yaml
dataset:
  source: csv
  path: data/customers.csv
  target_column: subscribed
  id_columns: [customer_id]
  task: binary_classification
  categorical_cardinality_limit: 100
  allow_high_cardinality: false
```

Files are content-addressed. BenchForge validates paths, columns, binary targets, dtypes, feature
availability, and categorical cardinality. Numeric columns use optional median imputation and
standardization. Categorical columns use most-frequent imputation and one-hot encoding with unknown
category handling. A fresh sklearn pipeline is fitted inside every inner or ordinary training fold;
learned preprocessing state is never shared across folds.

## Models, metrics, and search spaces

Supported classifiers are Logistic Regression, Random Forest, Extra Trees, Histogram Gradient
Boosting, and DummyClassifier. Supported metrics are accuracy, balanced accuracy, F1, and ROC-AUC.
Metric metadata records optimization direction; all current metrics are higher-is-better.

Versioned curated search spaces cover:

- Logistic Regression: log-scaled `C`, compatible penalty, and class weight.
- Random Forest and Extra Trees: estimator count, depth, split/leaf sizes, feature sampling, and
  class weight.
- Histogram Gradient Boosting: learning rate, iterations, leaves, leaf size, L2, and depth.

DummyClassifier is a fixed baseline and has no search space. Tree estimators default to `n_jobs=1`.

## Failure and ranking semantics

Trial failures retain state, full/proposed parameters, exception type/message, and duration. A bad
trial does not end a study. If no trial completes for a family/fold, that family is surfaced as a
failure and other families continue. If every family fails, the command fails clearly.

The tuned-family leaderboard includes fixed baselines and aggregates only outer-fold results. It
reports the “top measured tuned family” or “highest measured nested-CV mean”; it does not claim
statistical significance.

## Artifacts and reproducibility

Single-run and fixed-benchmark artifact formats remain compatible with V0/V0.2. Search writes:

```text
artifacts/search_<UTC timestamp>_<fingerprint>/
├── search_config.json
├── metadata.json
├── dataset_summary.json
├── outer_folds.json
├── tuned_leaderboard.json
├── tuned_leaderboard.csv
├── failures.json
├── final_candidate.json
└── families/<identifier>/
    ├── summary.json
    ├── outer_predictions.csv
    └── outer_fold_<n>/
        ├── split_plan.json
        ├── trials.json
        ├── trials.csv
        ├── best_params.json
        └── outer_result.json
```

The search fingerprint includes resolved dataset identity, candidates/fixed parameters, versioned
search-space definitions, both split configurations, preprocessing, metrics, primary metric,
budgets, seed, and final-search settings. It excludes timestamps, runtimes, artifact location, and
a file path when content identity already captures the data. With fixed data content,
configuration, BenchForge/dependency versions, and seed, sequential proposal sequences, folds,
selected parameters, predictions, metrics, ranking, and fingerprint should be materially
equivalent. Runtime fields are intentionally excluded from deterministic comparisons.

See [the architecture document](docs/architecture.md) for dependency boundaries.

## Current limitations and roadmap

V0.3 search is single-process and supports binary classification with stratified K-fold only. It
does not implement regression, pruning, multi-objective search, arbitrary pipelines,
feature-selection search, full AutoML, XGBoost, LightGBM, CatBoost, neural networks, GPUs,
distributed workers, robustness suites, statistical significance testing, a persistent analytics
catalog, model deployment, or a web UI.

The suggested next major slice is regression support: lower-is-better metric specifications,
regression models, compatible split planning, and versioned regression search spaces. A broader
AutoML controller should remain above—not inside—the reusable search layer.
