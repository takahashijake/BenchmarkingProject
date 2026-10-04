# BenchForge

BenchForge is reproducible benchmarking infrastructure for classical machine learning. It turns a
strict declarative configuration into leakage-safe evidence that can be inspected and reproduced.

BenchForge V0.8 supports binary classification, multiclass classification, and regression through
five workflow levels:

- `benchforge run`: execute one concrete pipeline.
- `benchforge benchmark`: compare several concrete pipelines on shared folds.
- `benchforge search`: tune several model families under a finite budget and compare the tuning
  procedures using untouched, shared outer folds.
- `benchforge robustness`: assess fixed candidates with repeated matched CV, paired effects,
  and ranking stability.
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
benchforge robustness configs/examples/breast_cancer_robustness.yaml
benchforge robustness configs/examples/iris_multiclass_robustness.yaml
benchforge robustness configs/examples/diabetes_regression_robustness.yaml
```

The Python API exposes `load_run_config()` / `run_benchmark()`,
`load_benchmark_config()` / `run_benchmark_suite()`, and
`load_search_config()` / `run_search()`, and
`load_automl_config()` / `build_automl_plan()` / `run_automl()`, plus
`load_robustness_config()` / `run_robustness()`.

## Robustness: how stable is the comparison?

A single cross-validation score depends on the fold assignment and estimator randomness. A small
lead in one benchmark may reverse when those change. V0.7 compares **fixed candidate
configurations** in repeated matched CV: each repetition gets a new deterministic fold seed,
all candidates receive that exact fold plan, and the existing fold kernel fits a fresh pipeline
using training rows only. There is no tuning or adaptive candidate selection in this workflow.

```yaml
schema_version: 1
task: binary_classification
dataset: {name: breast_cancer}
split: {strategy: stratified_kfold, n_splits: 3, shuffle: true}
preprocessing: {median_imputation: true, standardize: true}
models:
  - {name: logistic_regression, id: logistic, parameters: {max_iter: 1000}}
  - {name: random_forest_classifier, id: forest, parameters: {n_estimators: 40}}
metrics: [roc_auc, balanced_accuracy, f1]
primary_metric: roc_auc
robustness:
  repetitions: 5
  confidence_level: 0.95
  bootstrap_samples: 2000
seed: 42
output: {directory: artifacts}
```

```bash
benchforge robustness configs/examples/breast_cancer_robustness.yaml
```

```python
from benchforge import load_robustness_config, run_robustness

result = run_robustness(load_robustness_config("robustness.yaml"))
print(result.leaderboard)
print(result.pairwise_comparisons)
print(result.rank_stability)
```

The examples use five repetitions so they are inexpensive demonstrations. Increase `repetitions`
(for example to 20 or 50) for a more informative assessment. Execution is sequential. Its
approximate maximum fit count is **repetitions × candidates × folds**: the three-candidate Breast
Cancer example requires 45 fits. Increasing bootstrap samples does not fit additional estimators.
Classification uses `stratified_kfold`; regression uses `kfold`. Metrics retain existing names,
including `root_mean_squared_error` and `mean_absolute_error` rather than RMSE/MAE aliases.
Multiclass predictions keep the existing absent scalar `score` convention.

Each candidate's repetition score is the ordinary unweighted mean of its fold metrics. Its
robustness score is the mean across completed repetition scores. The reported standard deviation
is the **sample standard deviation across repetitions**, distinct from fold-score variability.
All robustness summary standard deviations use `ddof=1`; a single available observation has a
descriptive standard deviation of zero, with its count always visible.

Every unordered candidate pair has explicit paired evidence. `raw_paired_deltas` records
A minus B in natural metric units. `paired_deltas` reverses that sign for minimized metrics:
**positive always means A performed better than B**, and negative means B performed better.
Win rates count strictly positive/negative deltas; exact zero is a tie. Comparisons use only
repetitions where both candidates completed all folds, and retain the usable repetition IDs.

The robustness interval is a deterministic **percentile bootstrap of the mean paired delta**:
resample complete repetition-level deltas with replacement, calculate each resampled mean, and
use the two equal-tail quantiles at the configured confidence level. NumPy PCG64 uses a
SHA-256-derived pair-specific seed; linear quantiles and the sample count are versioned semantics.
`bootstrap_samples` defaults to 2,000 and accepts 100–100,000. Temporary sampling arrays are
batched. Fewer than two pairs produces an unavailable interval, not an invented bound.

An interval such as mean delta `+0.008`, 95% robustness interval `[-0.002, +0.017]` indicates
that the apparent advantage is sensitive to this resampling experiment. These intervals describe
**fold and estimator randomness conditional on the observed dataset**. Repetitions reuse rows;
they are not independent datasets. They do not estimate population uncertainty, prove superiority,
or establish statistical significance. BenchForge performs no p-value or fold-level hypothesis
test. A zero-crossing interval alone also does not establish practical equivalence; that would
require a task-specific effect threshold, which V0.7 does not define.

Rank stability summarizes competition ranks per repetition: exact ties share rank (for example
`1, 1, 3`). Identifiers only order presentation, never break a measured tie. Mean/median rank,
rank standard deviation, best/worst rank, first-place and top-two counts/rates, and full rank
frequencies are exposed. First-place and top-two rates use **all configured repetitions** as the
denominator, including failures; tied candidates can each count as first, so first-place rates
need not sum to one. Mean ranks use the repetitions where that candidate was ranked. Top-two is
reported even with two candidates, when it is simply an availability measure.

The leaderboard puts fully completed candidates before partial candidates, followed by candidates
that never completed a repetition. Within each group it orders by primary-metric mean in the
metric's optimization direction, then identifier. The highest mean score, highest first-place
rate, and lowest rank variability describe different properties; none proves a definitive winner.
The report gives paired evidence for every pair and conservative interpretations. A “consistent
advantage” description requires at least 90% strict wins and an interval entirely on that
candidate's side of zero; this reporting heuristic is not a hypothesis test. Exact all-repetition
ties are reported as no measured separation.

A failed candidate stops its current repetition at the failing fold; other candidates and later
repetitions continue. Completed fold metrics/predictions, the failed fold, skipped folds, and
exception details are retained. Incomplete repetitions do not contribute aggregate scores or
paired effects. Ranks compare the available successful candidates, which may bias comparisons
when failures occur; the report flags this. Even pairs with zero observations remain present.
If every candidate fails every repetition, artifacts are written before
`RobustnessExecutionError` is raised; its `.result` preserves all evidence and the artifact path.
The CLI exits with code 2 in that case. Partial runs report their failures and exit successfully.

Robustness artifacts have their own schema version 1:

```text
artifacts/robustness_<UTC timestamp>_<fingerprint>/
├── manifest.json
├── robustness_config.json
├── metadata.json
├── dataset_summary.json
├── repetition_plan.json
├── robustness_leaderboard.json / .csv
├── pairwise_comparisons.json
├── rank_stability.json
├── failures.json
└── repetitions/repetition_000/
    ├── split_plan.json
    ├── metrics.json
    ├── ranks.json
    └── candidates/<identifier>/
        ├── summary.json  # includes partial fold evidence and failure details
        └── predictions.csv
```

`benchforge.storage.read_robustness_artifacts()` reads the complete experiment, including failed
summaries and partial predictions. Fold assignments, repetition metrics, paired deltas, bootstrap
seeds/settings, and ranks are sufficient to reconstruct every reported comparison.

The robustness fingerprint includes content-addressed dataset identity, fixed candidates,
preprocessing, split/metric/primary-metric settings, repetitions, confidence/bootstrap settings,
seed, methodology version, model capabilities, and relevant dependency versions. Candidate and
metric ordering, output location, source paths superseded by content identity, timestamps, and
runtime do not affect identity. Repetition folds and predictions reproduce for the same content,
configuration, versions, and seed. Changing the fold seed does not guarantee a different partition
on a tiny dataset, though repetition seeds are distinct. Stochastic estimators also receive
candidate/repetition-specific seeds, so this experiment measures both sources of randomness.

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
implemented in V0.7, and multiclass prediction artifacts intentionally leave the scalar `score`
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
├── manifest.json
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

V0.7 search and AutoML are single-process. AutoML does not invent pipelines or adapt family budgets
from outer-fold evidence. BenchForge does not implement multiclass ROC-AUC, pruning,
multi-objective search, arbitrary pipelines, feature-selection or preprocessing search, XGBoost,
LightGBM, CatBoost, neural networks, GPUs, distributed workers, statistical
significance testing, a persistent analytics catalog, model deployment, or a web UI. Adaptive
racing may be considered only if it can operate wholly inside outer-training partitions.

## V0.8: persistent experiment catalog and artifact verification

The local catalog indexes immutable experiment evidence; it does not replace artifact directories.
Keep those directories available. SQLite stores identities, candidate/metric summaries, physical
locations and verification records, rather than predictions or trial histories. No database server
or training is needed to browse or verify evidence.

```bash
benchforge search configs/examples/breast_cancer_search.yaml
benchforge robustness configs/examples/breast_cancer_robustness.yaml

# Use the actual artifact directory printed by each command.
benchforge verify artifacts/search_<timestamp>_<fingerprint> --deep
benchforge verify artifacts/robustness_<timestamp>_<fingerprint> --json
benchforge catalog ingest artifacts/
benchforge catalog list --type search
benchforge catalog list --task regression --metric root_mean_squared_error
benchforge catalog show <experiment-id>
benchforge catalog verify <experiment-id> --deep
benchforge catalog compare <experiment-id-a> <experiment-id-b> --json
```

The default database is `.benchforge/catalog.sqlite3` relative to the current directory. Override it
with `benchforge catalog --database /path/history.sqlite3 list` (the option also works after the
catalog action). Experiment IDs are stable SHA-256 identifiers of workflow type plus semantic
fingerprint; an unambiguous ID prefix works for `show`, `verify` and `compare`. Filters for `list`
are `--type`, `--task`, `--dataset` (exact persisted dataset identity), `--metric`, and `--status`.
All new commands support `--json`; historical queries use stable ordering. `show` and `list` display
the **last recorded** verification, while `catalog verify` performs a fresh check of every copy.

**Semantic fingerprint:** “Are these experiment semantics the same?” It excludes artifact output
paths, source paths superseded by content identity, timestamps, runtime and metric presentation
ordering. Built-in datasets retain their versioned sklearn identities; local files retain their
content-based identities. `RunConfig.fingerprint` remains the unresolved configuration identity
for API compatibility; V0.8 `RunResult.fingerprint` uses resolved dataset identity. Benchmark,
search and AutoML fingerprints now ignore metric order. Old fingerprints are preserved when reading
historical evidence; the catalog does not retroactively merge different fingerprint algorithms.

**Artifact manifest:** “Are these persisted bytes the evidence that was originally sealed?” Every
new experiment has `manifest.json`, written last after structural/semantic checks. Manifest schema 1
protects **every regular file recursively**, including child manifests, except the experiment's own
root manifest. Paths are relative, inventories are ordered, and each file has a SHA-256 digest. The
manifest includes the existing semantic fingerprint; file hashes do not redefine that fingerprint.
Runtime metadata is protected evidence but does not affect semantic identity. Copies retain their
identity and seal. Symlinks and non-regular members are rejected. A manifest detects byte changes
relative to its seal; it is not a signed provenance attestation against someone rewriting both
files and manifest.

Verification checks required files, JSON/CSV structure, supported schema versions, configured
candidate coverage, failure records, metric direction, aggregate consistency and nested evidence.
It checks fold membership and inner/outer split separation when indices are stored. `--deep` also
recomputes fold scores from predictions and robustness bootstrap summaries without retraining.
Verification prints `PASS`, `WARNING` or `FAIL` diagnostics. Warnings alone exit 0; verification
failures exit 2 and identify changed/missing files. An incomplete writer cannot publish a valid seal.
A V0.8 directory missing its required manifest fails verification.

Pre-V0.8 artifacts remain readable and catalogable. They report **legacy/unsealed**, with a warning
that cryptographic integrity cannot be established. Structural and redundant semantic evidence can
still be checked; historical search-space/model identity inputs that were never persisted cannot
be reconstructed authoritatively. Ingestion establishes an additional byte snapshot, so subsequent
catalog verification detects changes to indexed legacy evidence too.

Ingestion verifies the full batch deeply, then commits it in one transaction. Repeating ingestion
is idempotent. Copies or reproductions with the same workflow and fingerprint map to one experiment
with multiple locations; observations remain specific to each location. Conflicting semantics under
one fingerprint are an explicit identity-collision error, including legacy run fingerprints that
omitted dataset content. Altering evidence at an indexed location is an error: preserve the original
and ingest a new location. Moving a directory retains identity, but the old catalog location remains
visible as unavailable until evidence is restored there; V0.8 does not silently rewrite history.

Historical comparison classifies evidence as **compatible**, **partially compatible** or
**incompatible**. Different tasks, dataset identities, primary metrics/directions or workflows are
incompatible. Different evaluation settings, stored fold assignments, versions, candidate pools or
missing/failed observations are partial compatibility. Only shared candidate identifiers with the
same family and mode receive descriptive metric/rank/parameter differences. Runs have no declared
primary metric and therefore have limited comparison support. Error metrics keep natural values;
the oriented B-minus-A difference is positive when B has the better measured score.

Comparison freshly verifies all copies first. It reports the selected locations explicitly and
flags replicas with differing measured observations. JSON output includes compact final-selection
or robustness summaries. Aggregate CV means are **not independent hypothesis-test samples**. A
historical difference does not establish statistical significance, population superiority or a
winning deployment choice. Unavailable measurements remain unavailable.
