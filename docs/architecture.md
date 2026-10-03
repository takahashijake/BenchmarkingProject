# BenchForge V0.4 architecture

BenchForge separates atomic execution, fixed comparison, and search orchestration:

```text
CLI
├── run ─────────────────────────────────────────┐
├── benchmark -> fixed candidate orchestration ──┤
└── search -> nested-search orchestration         │
             ├── one shared outer fold plan      │
             ├── inner Optuna studies ───────────┤
             ├── selected outer refits ──────────┤
             └── optional final full-data search │
                                                 v
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

- `core` owns strict run, benchmark, and search configurations and semantic serialization.
- `data` owns built-in/file loading, validation, schema inference, and content identity.
- `splits` owns task-aware deterministic flat and nested fold plans in original dataset index
  space: stratified K-fold for binary classification and K-fold for regression.
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
- `storage` persists run, suite, and nested-search evidence.
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
objective scores, or parameter selection. Sentinel regression tests structurally prove that no
outer-validation sample enters inner training or validation.

This boundary is identical for both tasks. Classification nests stratified folds; regression nests
K-fold partitions. In both cases inner indices are mapped back to global dataset index space.

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
evidence, and optional final selection. JSON/CSV is sufficient for V0.4; no Optuna database is
required.

## Deliberately deferred

Multiclass classification, pruning, multi-objective optimization, arbitrary pipeline generation,
feature/preprocessing search, full AutoML policy, XGBoost, LightGBM, CatBoost, neural networks,
GPU/distributed scheduling,
robustness suites, statistical significance, deployment, persistent analytics, and web UI remain
future work.
