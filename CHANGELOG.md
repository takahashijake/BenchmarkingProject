# Changelog

## Unreleased — v0.10 checkpoint recovery slice

- Add opt-in durable candidate checkpoint workspaces for fixed benchmarks.
- Validate dataset, fold plan, model configuration, and versioned run identity before reuse.
- Persist completed candidates with atomic JSON writes, OS-backed locking, and SHA-256 corruption detection.
- Add `--workspace` and `--resume` CLI options and report reuse and re-evaluation decisions.
- Add process-death, corruption, concurrency, artifact, and deterministic-equivalence tests.
- Extend CI to verify dependency consistency and build distributions on Python 3.12/3.13.
- Document recovery boundaries; search, robustness, and AutoML resume remain deferred.

## Unreleased — v0.9 initial execution slice

- Add opt-in candidate-level parallel execution to `benchmark` via `--workers` and Python `workers=`.
- Isolate candidates in processes, cap in-worker numerical-library threads and reject nested `n_jobs` oversubscription.
- Preserve configured candidate order and shared CV fold identities.
- Add equivalence, failure-isolation, and argument validation tests.
- Document benchmarking commands, resource behavior, and deferred resume/atomic-commit work.
