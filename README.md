# BenchForge

BenchForge is reproducible benchmarking infrastructure for classical machine learning. It turns a
declarative configuration into leakage-safe cross-validation evidence that can be inspected,
compared, and reproduced later.

BenchForge V0.2 supports two related workflows:

- `benchforge run` executes one controlled model experiment.
- `benchforge benchmark` compares several model families on one dataset using exactly the same
  validation folds and produces a measured leaderboard.
- Future AutoML will generate and search candidate pipelines using this same experiment kernel; it
  is not implemented yet.

## Installation

BenchForge requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run the examples

Run the original single-model experiment:

```bash
benchforge run configs/examples/breast_cancer_logreg.yaml
```

Compare five model families on the built-in offline dataset:

```bash
benchforge benchmark configs/examples/breast_cancer_suite.yaml
```

Run a mixed numerical/categorical user-file example:

```bash
benchforge benchmark configs/examples/mixed_tabular_suite.yaml
```

The Python API exposes `load_run_config()` / `run_benchmark()` and
`load_benchmark_config()` / `run_benchmark_suite()` for programmatic execution.

## User datasets

CSV and Parquet sources are configured explicitly:

```yaml
dataset:
  source: csv                 # csv or parquet
  path: data/customers.csv
  target_column: subscribed
  id_columns: [customer_id]   # excluded from features
  task: binary_classification
  categorical_cardinality_limit: 100
  allow_high_cardinality: false
```

Files are read locally; no network is required. BenchForge rejects missing files, missing target or
ID columns, duplicate columns, missing targets, non-binary targets, empty feature sets, unsupported
dtypes, and categorical columns above the configured cardinality limit. Set
`allow_high_cardinality: true` only after deciding one-hot expansion is appropriate.
Relative input paths are resolved from the process working directory, so repository examples are
intended to be run from the repository root.

Numeric pandas dtypes are numerical. Boolean, string, object, and pandas categorical dtypes are
categorical; booleans are intentionally one-hot encoded. Datetime and arbitrary unsupported dtypes
are rejected. String binary labels are encoded deterministically in sorted order, and the original
label order is recorded in the dataset summary. Free-form text modeling is outside V0.2.

## Leakage-safe mixed preprocessing

Every fold receives a new sklearn `Pipeline` and `ColumnTransformer`. Numeric columns use optional
median imputation and standardization. Categorical columns use most-frequent imputation and
one-hot encoding with `handle_unknown="ignore"`, so validation-only categories do not fail.

All imputer, scaler, and encoder state is learned exclusively from training-fold rows. No learned
transformation is fitted to the complete dataset. Histogram gradient boosting declares a dense
input requirement through model metadata; other models may retain sparse one-hot output.

## Models and metrics

Supported classifiers:

- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- Dummy Classifier baseline

Supported metrics are accuracy, balanced accuracy, F1, and ROC-AUC. All currently have a
higher-is-better direction. Estimator randomness comes from the benchmark seed, and parallel tree
defaults are conservative (`n_jobs=1`).

## Fair comparison and leaderboard

The suite runner resolves and validates the dataset once, creates one deterministic stratified fold
plan, and passes that immutable plan to the existing single-experiment kernel for every candidate.
This paired evaluation makes model scores comparable on the same held-out samples.

Successful candidates are ranked by descending mean primary metric, with deterministic model-ID
tie-breaking. The leaderboard includes mean, population standard deviation, all requested metrics,
runtime, and status. Failed candidates remain visible but receive no rank. Output describes the
leader as the highest measured mean in this benchmark; it does not claim statistical superiority.

## Artifacts and reproducibility

Single experiments retain the V0 JSON/CSV artifact format. A suite creates:

```text
artifacts/benchmark_<UTC timestamp>_<fingerprint>/
├── benchmark_config.json
├── metadata.json
├── dataset_summary.json
├── leaderboard.json
├── leaderboard.csv
├── failures.json
└── runs/
    └── <model identifier>/
        ├── config.json
        ├── metadata.json
        ├── fold_metrics.json
        ├── aggregate_metrics.json
        └── predictions.csv
```

The suite fingerprint covers normalized benchmark semantics and the resolved dataset identity. User
files are content-addressed with SHA-256, so changing file contents changes suite identity; moving an
identical input file does not. Model order, timestamps, output paths, machine paths, and measured
runtimes are excluded. Model execution order therefore does not change suite identity.

Exact runtime is naturally machine-dependent. With the same data content, configuration,
BenchForge/dependency versions, and seed, fold assignments, predictions, metrics, ranking, and
fingerprint are expected to be materially identical.

See [the architecture document](docs/architecture.md) for dependency boundaries.

## Current limitations

BenchForge V0.2 is single-process and supports binary classification with stratified K-fold only.
It does **not** implement regression, hyperparameter optimization, an AutoML controller, XGBoost,
LightGBM, CatBoost, text modeling, robustness sweeps, statistical significance testing, distributed
execution, an experiment database, or a web UI.

## Roadmap

**NOW**

- user-supplied mixed tabular datasets
- shared-fold multi-model benchmark suites
- reproducible leaderboards and suite evidence

**NEXT**

- richer schemas and regression
- persistent experiment/run catalog
- budgeted hyperparameter optimization
- statistical model comparison

**LATER**

- AutoML controller
- robustness suites
- parallel/distributed execution
- richer reporting and UI
