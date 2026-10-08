# Changelog

## Unreleased — v0.9 initial execution slice

- Add opt-in candidate-level parallel execution to `benchmark` via `--workers` and Python `workers=`.
- Isolate candidates in processes, cap in-worker numerical-library threads and reject nested `n_jobs` oversubscription.
- Preserve configured candidate order and shared CV fold identities.
- Add equivalence, failure-isolation, and argument validation tests.
- Document benchmarking commands, resource behavior, and deferred resume/atomic-commit work.
