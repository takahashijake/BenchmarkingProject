# Writing and using providers (interface v1)

BenchForge v0.12 adds opt-in extensions for datasets, sklearn estimators, scalar
metrics and Optuna search spaces. The scientific evaluator, preprocessing,
splitting and aggregation remain the same for built-ins and extensions.

## Install and run the independent example

From a BenchForge checkout, with Python 3.12 or 3.13:

```bash
python -m pip install -e '.[dev]'
python -m pip install ./examples/provider
benchforge plugins list
benchforge plugins inspect example --version 1.0.0
benchforge benchmark configs/providers/benchmark.yaml
benchforge search configs/providers/search.yaml
benchforge benchmark configs/providers/benchmark.yaml --workspace artifacts/recovery/provider
benchforge benchmark configs/providers/benchmark.yaml --resume artifacts/recovery/provider
benchforge verify <Artifacts-path-printed-by-the-command> --deep
benchforge catalog ingest <Artifacts-path-printed-by-the-command>
```

The provider package is independently installable, declares its own dependencies
and entry point, and never edits BenchForge registries. Installing a BenchForge
wheel and then the provider package works the same way. `scripts/wheel_smoke.py`
exercises installed-wheel execution outside the source checkout; CI runs it on
both supported Python versions.

`plugins list` reads installed distribution metadata, without importing provider
modules. `inspect` explicitly imports and initializes the requested, pinned
provider and validates its interface. Neither command uses the network. A broken
unselected provider cannot affect activation of another provider. Duplicate IDs
are errors; listing never silently chooses a winner.

Activation is explicit in each experiment:

```yaml
providers: [{id: example, version: 1.0.0}]
dataset: {name: example.toy}
models: [{name: example.logistic}]
metrics: [accuracy, example.agreement]
primary_metric: example.agreement
```

A single `run` uses `model:`; benchmark/search/robustness use `models:`. AutoML
uses `models: {include: [example.logistic]}` and its ordinary global budget.
The complete benchmark and search configurations live in `configs/providers/`.

## Python API

Existing entry points activate configuration selections automatically:

```python
from benchforge import load_benchmark_config, run_benchmark_suite
from benchforge.providers.runtime import discover, resolve
from benchforge.providers import ProviderRef

print(discover())  # metadata only
resolved = resolve((ProviderRef(id="example", version="1.0.0"),))
print(resolved.manifests)  # explicit trusted activation
result = run_benchmark_suite(load_benchmark_config("configs/providers/benchmark.yaml"))
print(result.leaderboard)
```

`resolve()` returns frozen `ResolvedProviders` with new dataset/model/search
registries and an immutable metric callback mapping. `activate(config)` returns
a resolved configuration copy and runtime. Caller configurations and built-in
registries remain unchanged. Existing custom-registry APIs remain available for
serial experiments; combining manual registries and provider activation is
rejected when the registries are different.

## Package contract

Declare a factory under the `benchforge.providers` entry-point group:

```toml
[project.entry-points."benchforge.providers"]
example = "my_package:provider"
```

The zero-argument factory returns `benchforge.providers.runtime.Provider`:

- `manifest`: frozen `ProviderManifest` with `id`, `api_version: 1`, and a pinned
  semantic implementation `version` such as `1.0.0`.
- `datasets`: mapping of component IDs to zero-argument `Dataset` loaders.
- `estimators`: mapping of IDs to existing `ModelSpec` objects.
- `metrics`: mapping of IDs to `(truth, prediction, score_or_none) -> float` callbacks.
- `search_spaces`: mapping of estimator IDs to existing `SearchSpace` objects.

All four mappings are required, and unused categories use `{}`. Callback keys
must match declarations exactly. IDs are lowercase ASCII segments with letters,
digits, underscores and hyphens; segments are separated by dots, start with a
letter, and the complete ID has at most 128 characters. Every component must
start with its provider ID followed by a dot. Unsafe names, duplicates,
unsupported interfaces and mismatched identities fail with actionable errors.
The working implementation is in
[`examples/provider/src/benchforge_example/__init__.py`](../examples/provider/src/benchforge_example/__init__.py).

### Datasets

A `Component` declares its ID and nonempty supported task tuple. Loaders return
the existing `Dataset`, including named feature groups, missing-value counts,
target metadata and a declared identity. Features and target require matching
consecutive row indices; target values must be finite. Classification requires
consecutive integer class IDs beginning at zero and matching target labels.
Numeric columns must be numeric and cannot contain infinity. Missing feature
values use the normal preprocessing policy. Data crosses the registry boundary
through deep copies of frames and metadata.

BenchForge supplements the declared identity with a SHA-256 digest of resolved
values, ordering, index, feature schema/groups, task, target name and class-label
metadata. Changing bytes while retaining a declared dataset version therefore
changes experiment identity too. The digest is a reproducibility identifier,
not a signed source-data attestation. Dataset loaders receive no training folds;
preprocessing is always constructed and fitted within the existing fold kernel.
V1 has no parameterized loader options: expose distinct versioned dataset IDs
when appropriate.

### Estimators

`EstimatorDeclaration` declares supported tasks and `requires_dense`, and must
match its `ModelSpec`. Factories accept a private parameter dictionary and the
centrally derived seed and return a sklearn `BaseEstimator`. BenchForge clones
the result at each construction, checks classifier/regressor task tags, and sets
`random_state` to the central seed when that parameter exists. Factories should
honor the seed for other randomness too. They must not share mutable fitted
state or change global registries. Standard sparse/dense preprocessing follows
`ModelCapabilities(requires_dense=...)`. Plugin factory failures remain visible
candidate/fold failures, following the workflow's existing failure policy.

### Metrics

`MetricDeclaration` declares tasks, direction (`maximize` or `minimize`), scalar
score requirement, display name, `semantics_version`, and validation behavior
`finite-scalar-v1`. Prediction callbacks work for binary classification,
multiclass classification and regression. Score callbacks currently support
binary classification only and receive the normal positive-class scalar score
from `predict_proba` or `decision_function`. Multiclass probability matrices are
outside v1. Unsupported combinations fail before fitting. Callbacks receive
private array copies and must return a finite real scalar; booleans, arrays,
complex values, NaN and infinity are rejected. Custom metric names do not extend
the `MetricName` enum: built-in enum members remain unchanged, while
`ExternalMetric`/`parse_metric` support namespaced identifiers.

### Search spaces

`SpaceDeclaration` uses the estimator ID, an integer space version, and canonical
JSON objects in `definition_json` and `defaults_json` (default `{}`). Their
values must match the actual `SearchSpace` definition and defaults. Canonical
JSON has sorted keys, compact separators and finite values. Proposal functions
receive a private copy of fixed parameters. Any proposed fixed key is rejected;
user-fixed values cannot be overwritten. Existing sequential seeded Optuna TPE,
trial allocation, inner/outer leakage boundaries and budgets are unchanged.

## Identity, artifacts and recovery

All explicitly activated providers and their complete scientific declarations
participate in canonical experiment configuration, including implementation,
interface, metric-semantics and search-space versions. This conservative v1
contract includes all declarations of each selected provider, even if only one
component is used. Changing an unused declaration in an activated provider can
therefore invalidate equivalence; merely installing an unselected package does
not. Provider/component declaration order and supported-task order are canonical.
Output directories and discovery order are excluded. Empty extension fields are
omitted, preserving historical built-in-only fingerprints.

`provider_schema_version: 1`, resolved `provider_manifests`, and
`provider_distributions` are recorded in sealed machine-readable metadata.
Distribution name, installed version and entry-point target are audit provenance,
separate from the provider's declared scientific version and experiment config.
A distribution-only packaging change does not change scientific identity.
Authors must bump declared scientific versions when behavior changes. Metadata
cannot prove an honest version declaration or bit-for-bit reproducibility across
hardware, dependencies and environments.

Serial benchmark checkpoints reuse the existing schema, atomic writer,
workspace lock, JSON result adapter and compatibility hash. Changed provider
semantics, dataset content/identity, folds, configuration or evaluator dependencies
invalidate reuse. Missing or corrupt candidate metadata triggers recomputation;
an incompatible workspace is rejected. Failed candidates never become cached
successes. Legacy built-in-only checkpoints remain readable.

Offline artifact verification and catalog ingestion never discover or import
providers. They validate metadata schema, version pins, provenance coverage,
identifiers, hashes and consistency. Built-in scores can still be recomputed.
Custom scores receive structural and aggregate checks, plus an explicit warning
that their callback semantics were **not recomputed**. A seal detects changes
relative to its inventory; it is not a signature against an attacker rewriting
both evidence and manifest.

## Execution limits and trust

| Workflow | Activated providers | Execution policy |
| --- | --- | --- |
| run | All four categories as relevant | Ordinary serial CV |
| benchmark | Dataset, estimator, metric | Serial; checkpoint/resume supported |
| search | All four | Existing serial nested search |
| robustness | Dataset, estimator, metric | Existing serial repeated matched CV |
| automl / plan-only | All four | Existing deterministic planner and search |
| benchmark workers > 1 or hard execution budgets | Rejected before activation | No guarantee is silently downgraded |

An eventual provider-ID worker bootstrap must reconstruct selected code in
workers and validate declarations against the coordinator. V1 deliberately does
not pickle resolved plugin objects or load executable checkpoint data. No memory
limits or sandbox isolation are introduced. Activated plugins are trusted Python
executable code with your process permissions; install and activate only code
you trust. Provider code must not mutate shared state, launch unmanaged processes,
or perform scientifically hidden preprocessing.
