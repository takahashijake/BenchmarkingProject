# BenchForge

BenchForge is reproducible benchmarking infrastructure for classical machine learning. It turns a
strict declarative configuration into leakage-safe evidence that can be inspected and reproduced.

BenchForge V0.6 supports binary classification, multiclass classification, and regression through
four workflow levels:

- `benchforge run`: execute one concrete pipeline.
- `benchforge benchmark`: compare several concrete pipelines on shared folds.
- `benchforge search`: tune several model families under a finite budget and compare the tuning
  procedures using untouched, shared outer folds.
- `benchforge automl`: deterministically choose supported families and translate a global trial
  budget into an ordinary nested-search configuration.

AutoML is compute-aware orchestration over the existing search layer, not automatic pipeline
invention.

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
benchforge run configs/examples/iris_multiclass_run.yaml
benchforge benchmark configs/examples/iris_multiclass_suite.yaml
benchforge search configs/examples/iris_multiclass_search.yaml
benchforge automl configs/examples/iris_multiclass_automl.yaml --plan-only
benchforge automl configs/examples/iris_multiclass_automl.yaml
benchforge run configs/examples/diabetes_regression_run.yaml
benchforge benchmark configs/examples/diabetes_regression_suite.yaml
benchforge search configs/examples/diabetes_regression_search.yaml
benchforge automl configs/examples/breast_cancer_automl.yaml --plan-only
benchforge automl configs/examples/breast_cancer_automl.yaml
benchforge automl configs/examples/diabetes_regression_automl.yaml
benchforge run configs/examples/rental_prices_regression_run.yaml
```

The Python API exposes `load_run_config()` / `run_benchmark()`,
`load_benchmark_config()` / `run_benchmark_suite()`, and
`load_search_config()` / `run_search()`, and
`load_automl_config()` / `build_automl_plan()` / `run_automl()`.

## AutoML configuration and policy

AutoML accepts a global trial budget and creates a normal `SearchConfig` before delegating all
training and evaluation to `run_search()`:

```yaml
schema_version: 1
task: binary_classification
dataset: {name: breast_cancer}
outer_split: {strategy: stratified_kfold, n_splits: 5, shuffle: true}
inner_split: {strategy: stratified_kfold, n_splits: 3, shuffle: true}
preprocessing: {median_imputation: true, standardize: true}
metrics: [roc_auc, balanced_accuracy, f1]
primary_metric: roc_auc
budget:
  total_trials: 240
  final_search_trials: 40
  # timeout_seconds_per_outer_fold: 600
models:
  include: []
  exclude: []
  include_fixed_baselines: true
seed: 42
output: {directory: artifacts}
```

Candidates come from `ModelRegistry` and `SearchSpaceRegistry`. An empty `include` considers every
task-compatible registered model; a nonempty `include` restricts that set, and `exclude` removes
names. Unknown, task-incompatible, duplicate, or overlapping policy entries fail before training.
Task-compatible models with search spaces are tuned; models without spaces are fixed candidates
when `include_fixed_baselines` is enabled. Candidate order is canonical by model name.

For `S` searchable families, `O` outer folds, total trials `T`, and final reserve `F`, the planner
uses `floor((T - F) / (S × O))` trials for every family/fold study. The remainder is reported but
not adaptively reassigned. A plan that cannot fund at least one trial for every study is rejected.
`F = 0` disables final search. A configured timeout can stop studies early, so trial counts then
become upper bounds.

The plan reports an approximate maximum estimator-fit count:

```text
allocated outer trials × inner folds
+ one outer refit per candidate/fold
+ final-search trials × inner folds
```

Use `--plan-only` to inspect candidate names, allocation, fingerprints, and this estimate without
fitting a model. No outer-validation score influences candidate selection or budget allocation.

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

BenchForge includes the offline sklearn Breast Cancer Wisconsin binary-classification, Iris
multiclass-classification, and Diabetes regression datasets, and reads local CSV or Parquet:

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

Binary file targets must contain exactly two classes. Multiclass file targets must contain at least
three classes. Both are encoded to integer IDs using a deterministic ordering independent of row
order, while human-readable labels remain in dataset summaries. BenchForge never infers multiclass
semantics from a binary task declaration. For regression files, use `task: regression`; targets
must be numeric, finite, non-missing, and non-constant. Integer-valued regression targets remain
continuous and are never class encoded. Files are content-addressed. BenchForge validates paths,
columns, targets, dtypes, feature availability, and categorical cardinality. Numeric columns use
optional median imputation and standardization. Categorical columns use most-frequent imputation
and one-hot encoding with unknown category handling. A fresh sklearn pipeline is fitted inside
every inner or ordinary training fold; learned preprocessing state is never shared across folds.

## Tasks, models, metrics, and search spaces

Binary and multiclass classification use deterministic stratified K-fold and share the Logistic
Regression, Random Forest, Extra Trees, Histogram Gradient Boosting, and DummyClassifier families.
Binary metrics are ROC-AUC, F1, accuracy, and balanced accuracy. Multiclass metrics are macro F1,
accuracy, and balanced accuracy. All classification metrics are maximized. Binary `f1` keeps its
positive-class semantics; multiclass uses the explicit `f1_macro` metric. Multiclass ROC-AUC is not
implemented in V0.6, and multiclass prediction artifacts intentionally leave the scalar `score`
field empty.

Regression uses shuffled deterministic K-fold and supports Linear Regression, Ridge, Random
Forest, Extra Trees, Histogram Gradient Boosting, and DummyRegressor. Its metrics are RMSE and MAE
(minimized) and R² (maximized). Natural error values are retained: an RMSE of `52.34` is optimized
as `52.34` with direction `minimize`, never negated.

Versioned curated search spaces cover:

- Logistic Regression: log-scaled `C`, compatible penalty, and class weight.
- Random Forest and Extra Trees: estimator count, depth, split/leaf sizes, feature sampling, and
  class weight.
- Histogram Gradient Boosting: learning rate, iterations, leaves, leaf size, L2, and depth.
- Ridge: broad log-scaled `alpha`.
- Regression Random Forest and Extra Trees: estimator count, depth, split/leaf sizes, and feature
  sampling.
- Regression Histogram Gradient Boosting: learning rate, iterations, leaves, leaf size, L2, and
  optional depth.

Classifier search spaces are reused across binary and multiclass tasks. DummyClassifier and
DummyRegressor are fixed baselines and have no search spaces. LinearRegression may also remain
fixed. Tree estimators default to `n_jobs=1`.

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

AutoML composes those artifacts rather than serializing search evidence again:

```text
artifacts/automl_<UTC timestamp>_<fingerprint>/
├── automl_config.json
├── plan.json
├── generated_search_config.json
├── metadata.json
├── result.json
└── search/
    └── ... ordinary SearchArtifactStore output ...
```

See [the architecture document](docs/architecture.md) for dependency boundaries.

## Current limitations and roadmap

V0.6 search and AutoML are single-process. AutoML does not invent pipelines or adapt family budgets
from outer-fold evidence. BenchForge does not implement multiclass ROC-AUC, pruning,
multi-objective search, arbitrary pipelines, feature-selection or preprocessing search, XGBoost,
LightGBM, CatBoost, neural networks, GPUs, distributed workers, robustness suites, statistical
significance testing, a persistent analytics catalog, model deployment, or a web UI. Adaptive
racing may be considered only if it can operate wholly inside outer-training partitions.
