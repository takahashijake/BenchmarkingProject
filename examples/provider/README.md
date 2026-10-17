# Independent BenchForge provider example

Install BenchForge, then `python -m pip install ./examples/provider` from the
BenchForge repository. This is a separate package with an installed Python entry
point, not an in-process registry edit. It uses a deterministic 60-row mixed-type
dataset, a sklearn logistic estimator, a scalar agreement metric and a small
Optuna search space. No external dataset or network is needed to execute it.

See [the provider authoring and demonstration guide](../../docs/providers.md) for
installation, discovery, CLI/API execution, search, artifact verification,
checkpoint reuse, version semantics and execution restrictions.
