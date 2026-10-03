# BenchForge V0.2 architecture

BenchForge separates one atomic experiment from orchestration across candidates:

```text
CLI
├── run ────────────────────────────────┐
└── benchmark -> BenchmarkRunner        │
                 ├── resolve dataset once
                 ├── infer schema once │
                 ├── create folds once │
                 └── for each model ───┤
                                       v
                              experiment kernel
                              ├── fresh preprocessor
                              ├── fresh estimator
                              ├── fit training fold only
                              ├── evaluate validation fold
                              └── return fold evidence
```

`run_benchmark()` remains the atomic execution unit. It can resolve its own dataset and folds, which
preserves the V0 API, or accept a resolved `Dataset` and immutable `Fold` tuple. The suite layer uses
the latter path. Training, prediction, scoring, and aggregation therefore have one implementation;
future search and AutoML can reuse the same kernel.

## Dependency boundaries

- `core` owns strict configuration and semantic serialization.
- `data` owns built-in and local-file loading, validation, schema inference, and dataset identity.
- `splits` owns deterministic fold plans.
- `preprocessing` builds fold-local sklearn transformers.
- `models` owns factories and minimal input-capability metadata.
- `evaluation` owns metrics and aggregation.
- `execution.runner` is the atomic single-model CV kernel.
- `execution.benchmark` owns multi-candidate orchestration, failures, timing, and ranking.
- `storage` persists single-run and suite artifacts.
- `analysis` formats completed results; the CLI contains no benchmark logic.

Registries are immutable instance-local mappings rather than mutable process globals. Dataset frames
are copied at registry boundaries and are not mutated during candidate execution.

## Why shared folds matter

Different random partitions can make a model look better simply because it received easier held-out
rows. A benchmark suite resolves the target once and constructs one seeded stratified split plan.
Every model receives the exact same train/validation index arrays. Out-of-fold records preserve the
sample index and fold ID, making the pairing auditable.

## Leakage boundary

For each model and fold, the experiment kernel constructs a new sklearn `Pipeline` containing a new
`ColumnTransformer` and estimator. Numeric imputation/scaling and categorical imputation/one-hot
vocabularies are fitted only by `pipeline.fit(train_x, train_y)`. Validation data is transformed only
after fitting. Tests explicitly verify that held-out numerical extremes and held-out categories do
not influence learned preprocessing state.

## Model capabilities

`ModelSpec` pairs a factory with a deliberately small `ModelCapabilities` value. V0.2 uses only
`requires_dense`; preprocessing chooses dense or sparse one-hot output before the pipeline is built.
This keeps estimator-specific conversion out of the execution loop without creating a speculative
capability framework.

## Identity and persistence

Single-run fingerprints retain V0 canonicalization. Suite fingerprints hash canonical suite
semantics plus resolved dataset identity. Model entries are sorted during fingerprinting because
execution order is not semantic. For file datasets, the content SHA-256 is semantic while the input
path is not. Output directory, timestamps, runtime, and machine-specific locations are excluded.

The suite artifact root contains normalized configuration, provenance, dataset schema/missingness,
JSON and CSV leaderboards, structured failures, and nested V0-compatible artifacts for every
successful candidate. This shape is intended to be directly ingestible by a future run catalog.

## Deliberately not implemented

Hyperparameter optimization, AutoML policy, regression, XGBoost, LightGBM, CatBoost, robustness
sweeps, distributed execution, statistical significance claims, and web UI remain future work.
