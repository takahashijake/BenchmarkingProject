# BenchForge

BenchForge is the beginning of a reproducible experimentation and AutoML platform for classical
machine learning. V0 deliberately implements one complete path instead of a broad collection of
partly connected features: deterministic, leakage-safe binary classification with stratified
cross-validation and inspectable local artifacts.

## Current V0 scope

Implemented today:

- strict Pydantic configuration loaded from YAML;
- a stable SHA-256 experiment fingerprint over canonical configuration;
- the offline scikit-learn Breast Cancer Wisconsin dataset;
- deterministic stratified K-fold plans;
- fold-local median imputation and standardization in sklearn pipelines;
- logistic-regression and random-forest classifier factories;
- accuracy, balanced accuracy, F1, and ROC AUC;
- fold metrics, aggregate mean/population standard deviation, and out-of-fold predictions;
- local JSON/CSV provenance artifacts and a concise CLI summary.

See [the architecture note](docs/architecture.md) for dependency boundaries and the leakage
invariant. The packages under `search`, `robustness`, and `automl` mark deliberate domain boundaries;
they do not claim those capabilities are implemented.

## Install and run

BenchForge requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
benchforge run configs/examples/breast_cancer_logreg.yaml
```

The random-forest example is available at
`configs/examples/breast_cancer_random_forest.yaml`. The Python API is equally direct:

```python
from benchforge import load_run_config, run_benchmark

result = run_benchmark(load_run_config("configs/examples/breast_cancer_logreg.yaml"))
print(result.aggregate_metrics)
```

## Artifacts and reproducibility

Each run creates `artifacts/<UTC timestamp>_<fingerprint prefix>/` containing:

- `config.json`: normalized validated configuration;
- `metadata.json`: dataset identity, full fingerprint, seed, creation time, and runtime versions;
- `fold_metrics.json`: metrics plus train/validation sizes for every fold;
- `aggregate_metrics.json`: mean and population standard deviation across folds;
- `predictions.csv`: sample index, fold, truth, prediction, and probability/decision score.

The top-level seed controls split shuffling and estimator randomness. A fresh sklearn preprocessing
and estimator pipeline is fit inside every training fold. Timestamps and artifact paths do not enter
the fingerprint. With the same configuration, dataset, BenchForge version, and dependency versions,
splits, predictions, and metrics are expected to be materially identical.

## Limitations

V0 supports one built-in numerical binary-classification dataset, stratified K-fold only, and local
single-process execution. It has no categorical schema, regression path, search, run database,
distributed execution, statistical comparison, robustness suite, or AutoML controller. CSV,
Parquet, and network dataset ingestion are intentionally deferred.

## Roadmap

**NOW**

- reproducible benchmark kernel

**NEXT**

- richer datasets and schemas
- broader model registry
- experiment/run database
- hyperparameter optimization

**LATER**

- AutoML controller
- robustness suites
- statistical benchmark comparison
- parallel/distributed execution
- richer reporting and UI
