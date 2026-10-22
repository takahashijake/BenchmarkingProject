# Changelog

## Unreleased — v0.15 verification-gated reporting

- Add an immutable public reporting model and adapters for all five persisted workflows.
- Generate deterministic, offline HTML with accessible tables and embedded CSS, without training.
- Preserve outer evaluation, inner/final selection, partial failures and exact robustness evidence.
- Gate reports on existing verification, support deep checks and require explicit legacy opt-in.
- Compare directories without a catalog; disclose compatibility and withhold insufficiently supported deltas.
- Escape metadata, reject unsafe destinations and publish atomically with explicit overwrite.
- Reject overflowing nonfinite JSON numbers at the existing verification decoder boundary.
- Add scientific, security, CLI and packaging regressions, Windows reporting checks and six examples.
- Preserve package/evaluator version 0.8.0, scientific identities and persisted schemas.

## Unreleased — v0.14 parallel search and AutoML

- Add bounded process execution with `--workers N` and Python `workers=N` (default 1).
- Plan immutable family/outer-fold tasks and reuse one scientific fold function in both modes.
- Reuse the subprocess supervisor with thread caps, crash isolation and canonical aggregation.
- Validate worker evidence and preserve coordinator-owned v0.13 recovery and atomic publication.
- Keep worker count outside scientific fingerprints, artifact schemas and workspace identity.
- Reject parallel study timeouts, providers, custom registries and nested `n_jobs` oversubscription.
- Support unbudgeted Windows subprocesses with direct-worker cleanup; retain POSIX group cleanup.
- Expose runtime-only task diagnostics, add real-process regression tests and Windows CI coverage.

## Unreleased — v0.13 durable search and AutoML

- Add opt-in `--workspace`/`--resume` and Python `checkpoint_dir`/`resume` interfaces.
- Recover validated complete family/outer-fold and final-search tasks through ordinary search.
- Bind recovery to resolved data, exact fold plans, scientific contracts, dependencies and AutoML plans.
- Reuse shared atomic JSON publication with OS locks, integrity checks and semantic validation.
- Support serial pinned providers; reject timeout-enabled recovery and unversioned registries.
- Preserve historical fingerprints, artifact schemas, fixed-benchmark behavior and catalog identities.
- Add real process-kill, scientific-equivalence, corruption, concurrency and identity-drift regressions.

## Unreleased — v0.12 versioned providers

- Add explicit, version-pinned installed providers for datasets, estimators, metrics and search spaces.
- Separate metadata discovery from trusted activation; add plugins list/inspect diagnostics.
- Integrate serial run, benchmark, search, robustness and AutoML with the existing evaluator.
- Include resolved scientific declarations and dataset content in identity; seal separate distribution provenance.
- Support serial provider checkpoint/resume and reject provider workers/hard budgets explicitly.
- Keep verification/catalog ingestion independent of provider execution; report custom-score verification limits.
- Preserve existing built-in catalog signatures when optional provider metadata is absent.
- Add a standalone example distribution, installed CLI/API tests and wheel smoke checks on both CI versions.
- Correct eager worker imports that caused baseline timeout regressions; preserve original timeout assertions.

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
