# BenchForge V0.7 architecture

BenchForge separates deterministic policy, search orchestration, fixed comparison, and atomic
execution:

```text
CLI
├── run
├── benchmark
├── search
├── robustness
│     ↓
│ deterministic repetition planner
│     ↓
│ shared fold plans (one per repetition)
│     ↓
│ existing execution kernel
│     ↓
│ paired comparison + rank stability analysis
└── automl
      ↓
  deterministic policy/planner
      ↓
  generated SearchConfig
      ↓
  existing nested-search orchestration
      ├── one shared outer fold plan
      ├── inner Optuna studies
      ├── selected outer refits
      └── optional final full-data search
          ↓
      experiment kernel
      ├── fresh preprocessor
      ├── fresh estimator
      ├── fit training rows only
      ├── evaluate validation rows
      └── return fold evidence
```

`run_benchmark()` remains the atomic CV execution unit and retains the V0 API. It may resolve its
own data/folds or accept a resolved `Dataset` and immutable fold tuple. Its extracted
`execute_fold()` operation performs the one final outer evaluation for a selected configuration.
Preprocessing, estimator construction, prediction, and scoring have one task-aware implementation;
there are no separate classification and regression runners.

## Dependency boundaries

- `core` owns strict run, benchmark, search, and AutoML configurations and semantic serialization.
- `automl` owns registry-driven candidate policy, deterministic budget allocation, and delegation
  to the search layer. It owns no fitting, scoring, splitting, preprocessing, or ranking logic.
- `data` owns built-in/file loading, validation, schema inference, and content identity.
- `splits` owns task-aware deterministic flat and nested fold plans in original dataset index
  space: stratified K-fold for binary and multiclass classification, and K-fold for regression.
- `preprocessing` builds fresh fold-local sklearn transformers.
- `models` owns estimator factories, supported-task metadata, and minimal dense-input capability
  metadata.
- `evaluation` owns metric task compatibility, score requirements, optimization direction, and
  fold aggregation.
- `execution.runner` is the atomic CV/fold kernel.
- `execution.benchmark` owns fixed multi-candidate orchestration.
- `search.spaces` and `search.registry` own versioned Optuna proposal semantics independently of
  the estimator registry.
- `search.runner` owns nested search, sequential studies, failures, outer aggregation, and final
  selection.
- `search.results` owns explicit immutable evidence types.
- `robustness.config` extends strict benchmark contracts for fixed repeated comparisons.
- `robustness.planner` owns canonical identity and SHA-256 repetition seed policy.
- `robustness.runner` passes shared folds to `execute_fold()` and preserves partial failures.
- `robustness.statistics` analyzes repetition-level paired effects and competition ranks.
- `robustness.results` owns frozen evidence classes with tuple-based metric/rank evidence.
- `core.seeds` contains the unchanged shared seed derivation; search retains its old import API.
- `storage` persists run, suite, search, AutoML, and versioned robustness evidence.
- `analysis` formats results; the CLI contains no execution logic.

Registries are immutable instance-local mappings. Dataset frames are copied at registry boundaries
and are not mutated during execution.

## Nested leakage boundary

The outer fold plan is created exactly once and shared by every family. For each outer fold,
`build_inner_folds()` splits only its training indices and maps relative split indices back to the
original dataset index space. Optuna objectives receive only these inner folds. Every inner fold
fits a new preprocessing pipeline on inner-training rows.

Outer-validation indices are recorded for audit but passed only to the post-selection
`execute_fold()` call. They cannot affect proposal, pruning (not implemented), preprocessing,
objective scores, or parameter selection. Sentinel regression and multiclass tests structurally
prove that no outer-validation sample enters inner training or validation.

This boundary is identical for all three tasks. Both classification tasks nest stratified folds;
regression nests K-fold partitions. In every case inner indices are mapped back to global dataset
index space.

Classification datasets preserve integer truth and prediction IDs through execution and artifacts.
Binary score-based metrics use the existing scalar positive-class score. Multiclass execution does
not apply that binary convention: V0.7 has no multiclass score-based metric and records `score` as
null while retaining target labels in dataset summaries.

The tuned-family leaderboard aggregates the exactly-once outer results. Inner objective scores
select parameters only. The optional final full-data score selects deployment parameters only.
Neither can rank families.

## Search spaces, budgets, and failures

`SearchSpaceRegistry` is keyed compatibly with `ModelRegistry`, while keeping proposal policy out of
model construction and orchestration. Every space has an explicit version and serializable
definition. User-fixed parameter names are omitted from proposals and merged without overwrite.

Each family/outer-fold study uses seeded TPE, sequential trials, and a deterministic estimator
seed. Trial count is the primary budget; optional timeout is a best-effort second stopping limit.
Expected parameter/value or numerical fold failures are recoverable trial failures; unexpected
exception types are not converted into objective scores. A fold with no completed trial fails its
family visibly, while unrelated families remain isolated.

## Identity and persistence

Search fingerprints hash canonical search semantics, resolved dataset content identity, and the
participating versioned search-space definitions. Output paths, source paths already replaced by
content identity, timestamps, durations, and machine locations are excluded.

Per-family/per-fold seeds use SHA-256 derivation instead of Python's randomized `hash()`. For fixed
data, configuration, versions, and seed, sequential TPE makes folds, proposals, selected
parameters, predictions, metrics, ranking, and semantic identity materially reproducible. Runtime
and timestamp fields are not deterministic.

Artifacts preserve shared outer folds, every nested split plan, complete/failed/pruned trial
records, selected inner parameters/scores, outer metrics and predictions, aggregate family
evidence, and optional final selection. JSON/CSV is sufficient for V0.7; no Optuna database is
required. AutoML adds a policy/config/plan wrapper and nests the unchanged search artifact format
under `search/`.

## AutoML planning boundary

The AutoML planner intersects each model's `supported_tasks` capability set with the configured
task and registered search spaces, optionally retains compatible no-space models as fixed
baselines, sorts all names, and creates `SearchModelConfig` values. The shared classifier entries
and search spaces serve both binary and multiclass tasks. The planner converts the global
outer-search budget with one uniform, precommitted allocation across every searchable family and
outer fold. Any integer remainder stays unused and visible. The planner never receives outer-fold
scores.

The AutoML fingerprint includes the policy excluding output paths, resolved dataset identity,
selected candidates and modes, versioned search-space identities, allocation semantics, and the
generated search fingerprint. Runtime, timestamps, and artifact paths remain outside identity.

## Robustness orchestration boundary

Robustness resolves the dataset once and precommits a deterministic plan. It sorts candidates by
identifier, derives distinct repetition fold seeds using the shared SHA-256 helper (resolving rare
32-bit seed collisions deterministically), and invokes the ordinary task-aware `build_folds()`
once per repetition. The exact same fold objects pass to every candidate's `execute_fold()` call.
Candidate/repetition estimator seeds are derived separately and recorded. No global random state
is introduced by robustness. Evaluation remains sequential with one fresh preprocessing pipeline
and estimator fitted only on each training fold.

Calling the fold primitive directly preserves successful earlier fold metrics/predictions when a
later fold fails, which the atomic all-or-nothing `run_benchmark()` result cannot expose. This is
orchestration over the shared kernel, not an independent CV implementation: metric computation,
aggregation, estimator creation, preprocessing, task validation, data loading, and split semantics
remain in their original packages. Successful repetition aggregation calls
`aggregate_fold_metrics()`. A failure ends only that candidate's current repetition and records
its failed/skipped fold IDs and root exception. Later repetitions still attempt every candidate.
All-failed experiments persist before raising an exception carrying the immutable result.

Analysis uses repetition-level fold means rather than treating overlapping CV folds as independent
observations. Every candidate pair retains available repetition IDs, natural A-minus-B deltas,
oriented better-than deltas, descriptive effect statistics, wins/losses/ties, and deterministic
percentile-bootstrap mean intervals. PCG64 bootstrap seeds, equal-tail linear quantiles, and sample
counts are explicit. Intervals require at least two pairs; missing pairs are never imputed.
These are conditional resampling-sensitivity intervals, not population inference or p-values.

Ranks use exact-score competition ties (1, 1, 3); candidate insertion order cannot affect ranks.
Rates use all planned repetitions; rank summaries use available successful observations and retain
missing counts. Failures can change the pool being ranked and are flagged in reports. The summary
leaderboard orders complete candidates before partial before failed, then direction-aware mean
and identifier. Comparison, rank variability, first-place frequency, and mean performance remain
separate concepts.

Robustness artifact schema 1 retains a top-level seed plan and each repetition's complete fold
indices, candidate fold/aggregate metrics, predictions (including partial failures), ranks, and
failure details. Existing workflow schemas remain unchanged. The reader loads all evidence, and
tests reconstruct pairs/ranks from it. Identity includes canonical config, resolved content
identity, model metadata, relevant versions, and a methodology version. Paths replaced by content
identity, output paths, duration, and timestamp are excluded. Reporting lives in
`analysis.robustness`; the CLI only loads, delegates, formats, and reports the artifact location.

## Deliberately deferred

Multiclass ROC-AUC, pruning, multi-objective optimization, arbitrary pipeline generation,
feature/preprocessing search, pipeline invention, XGBoost, LightGBM, CatBoost, neural networks,
GPU/distributed scheduling, statistical significance, deployment, persistent
analytics, and web UI remain future work. Outer-fold-driven family racing is explicitly excluded
because those folds remain unbiased evaluation evidence.
