from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic, perf_counter, process_time

from benchforge.core.config import BenchmarkConfig, ModelConfig
from benchforge.data.registry import (
    Dataset,
    DatasetRegistry,
    DatasetSummary,
    default_dataset_registry,
    summarize_dataset,
)
from benchforge.evaluation.metrics import AggregateMetric, metric_spec
from benchforge.execution.checkpoints import CheckpointWorkspace, _write_atomic
from benchforge.execution.runner import RunResult, run_benchmark
from benchforge.execution.supervisor import ExecutionOptions, supervise
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.splits.stratified import Fold, build_folds
from benchforge.storage.suite import BenchmarkArtifactStore


@dataclass(frozen=True)
class CandidateResult:
    model_identifier: str
    model_config: ModelConfig
    run_result: RunResult
    duration_seconds: float


@dataclass(frozen=True)
class CandidateFailure:
    model_identifier: str
    model_name: str
    error_type: str
    message: str
    duration_seconds: float


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int | None
    model_identifier: str
    status: str
    primary_metric_mean: float | None
    primary_metric_std: float | None
    metrics: dict[str, AggregateMetric]
    duration_seconds: float


@dataclass(frozen=True)
class ExecutionDiagnostic:
    model_identifier: str
    outcome: str
    elapsed_seconds: float
    reason: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    fingerprint: str
    dataset: DatasetSummary
    primary_metric: str
    fold_count: int
    candidates: tuple[CandidateResult, ...]
    failures: tuple[CandidateFailure, ...]
    leaderboard: tuple[LeaderboardEntry, ...]
    total_duration_seconds: float
    artifact_directory: Path | None = None
    checkpoint_workspace: Path | None = None
    checkpoint_reused: tuple[str, ...] = ()
    checkpoint_executed: tuple[str, ...] = ()
    checkpoint_invalid: tuple[str, ...] = ()
    execution_diagnostics: tuple[ExecutionDiagnostic, ...] = ()
    execution_options: ExecutionOptions | None = None


def _build_leaderboard(
    candidates: list[CandidateResult],
    failures: list[CandidateFailure],
    primary_metric: str,
    direction: str,
) -> tuple[LeaderboardEntry, ...]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            (
                -candidate.run_result.aggregate_metrics[primary_metric].mean
                if direction == "maximize"
                else candidate.run_result.aggregate_metrics[primary_metric].mean
            ),
            candidate.model_identifier,
        ),
    )
    successful = [
        LeaderboardEntry(
            rank=rank,
            model_identifier=candidate.model_identifier,
            status="success",
            primary_metric_mean=candidate.run_result.aggregate_metrics[primary_metric].mean,
            primary_metric_std=candidate.run_result.aggregate_metrics[primary_metric].std,
            metrics=candidate.run_result.aggregate_metrics,
            duration_seconds=candidate.duration_seconds,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]
    failed = [
        LeaderboardEntry(
            rank=None,
            model_identifier=failure.model_identifier,
            status="failed",
            primary_metric_mean=None,
            primary_metric_std=None,
            metrics={},
            duration_seconds=failure.duration_seconds,
        )
        for failure in failures
    ]
    return tuple([*successful, *failed])


def _execute_candidate(
    config: BenchmarkConfig,
    model: ModelConfig,
    dataset: Dataset,
    folds: tuple[Fold, ...],
    model_registry: ModelRegistry,
    limit_threads: bool,
) -> tuple[CandidateResult | None, CandidateFailure | None, float]:
    """One isolated candidate; return results without worker-side artifact writes."""
    from threadpoolctl import threadpool_limits

    identifier = model.id or model.name
    started = perf_counter()
    cpu_started = process_time()
    try:
        if limit_threads:
            with threadpool_limits(limits=1):
                result = run_benchmark(
                    config.to_run_config(model),
                    model_registry=model_registry,
                    persist=False,
                    dataset=dataset,
                    folds=folds,
                )
        else:
            result = run_benchmark(
                config.to_run_config(model),
                model_registry=model_registry,
                persist=False,
                dataset=dataset,
                folds=folds,
            )
        candidate = CandidateResult(identifier, model, result, perf_counter() - started)
        return candidate, None, process_time() - cpu_started
    except Exception as exc:
        failure = CandidateFailure(
            identifier, model.name, type(exc).__name__, str(exc), perf_counter() - started
        )
        return None, failure, process_time() - cpu_started


def _candidate_process_entry(
    arguments: tuple[BenchmarkConfig, ModelConfig, Dataset, tuple[Fold, ...]],
) -> tuple[CandidateResult | None, CandidateFailure | None, float]:
    config, model, dataset, folds = arguments
    return _execute_candidate(config, model, dataset, folds, default_model_registry, True)


def run_benchmark_suite(
    config: BenchmarkConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    persist: bool = True,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    execution_options: ExecutionOptions | None = None,
) -> BenchmarkResult:
    """Evaluate shared folds with optional verified candidate-level recovery."""
    started = perf_counter()
    options = execution_options or ExecutionOptions()
    deadline = monotonic() + options.run_timeout if options.run_timeout is not None else None
    if options.bounded and model_registry is not default_model_registry:
        raise ValueError("budgeted benchmark requires the default model registry")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires a checkpoint_dir")
    if checkpoint_dir is not None and model_registry is not default_model_registry:
        raise ValueError("checkpointing requires the versioned default model registry")
    dataset = dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError(
            f"dataset task '{dataset.task}' does not match configured task '{config.task}'"
        )
    folds = build_folds(dataset.target, config.split, config.seed, config.task)

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be at least 1")
    available_cpus = os.cpu_count() or 1
    if workers > available_cpus:
        raise ValueError(
            f"requested {workers} workers but only {available_cpus} logical CPUs detected"
        )
    if workers > 1 and model_registry is not default_model_registry:
        raise ValueError("parallel benchmark requires the default model registry")
    if workers > 1 or options.bounded:
        for model in config.models:
            jobs = model.parameters.get("n_jobs")
            if jobs is not None and jobs != 1:
                raise ValueError(
                    f"model '{model.id or model.name}' sets n_jobs={jobs!r}; "
                    "parallel candidates require n_jobs=1"
                )

    workspace = (
        CheckpointWorkspace(checkpoint_dir, config, dataset, folds, resume=resume)
        if checkpoint_dir is not None
        else None
    )
    # Only the coordinator commits checkpoints, never a worker. Completion order
    # does not influence final candidate order or scientific fold semantics.
    outcomes: dict[str, tuple[CandidateResult | None, CandidateFailure | None, float]] = {}
    with ExitStack() as stack:
        if workspace is not None:
            stack.enter_context(workspace)
        pending: list[ModelConfig] = []
        for model in config.models:
            identifier = model.id or model.name
            cached = workspace.load(model) if workspace is not None else None
            if cached is not None:
                outcomes[identifier] = (cached, None, 0.0)
            else:
                pending.append(model)

        outcome: tuple[CandidateResult | None, CandidateFailure | None, float]
        if (options.bounded or workers > 1) and os.name == "posix":
            arguments = [(config, model, dataset, folds) for model in pending]
            diagnostics = {
                model.id or model.name: ExecutionDiagnostic(
                    model.id or model.name, "not_started", 0
                )
                for model in config.models
            }
            for identifier in outcomes:
                diagnostics[identifier] = ExecutionDiagnostic(identifier, "reused", 0)
            launched: dict[str, float] = {}

            def on_start(index: int) -> None:
                identifier = pending[index].id or pending[index].name
                launched[identifier] = monotonic()
                diagnostics[identifier] = ExecutionDiagnostic(identifier, "running", 0)

            try:
                with closing(
                    supervise(
                        _candidate_process_entry,
                        arguments,
                        workers=workers,
                        options=options,
                        deadline=deadline,
                        on_start=on_start,
                    )
                ) as completions:
                    for completed in completions:
                        model = pending[completed.index]
                        identifier = model.id or model.name
                        if completed.reason is not None:
                            failure = CandidateFailure(
                                identifier,
                                model.name,
                                completed.reason,
                                "not started: run budget exhausted"
                                if completed.reason == "NotStarted"
                                else f"isolated candidate stopped: {completed.reason}; "
                                f"exit code {completed.exit_code}",
                                completed.elapsed,
                            )
                            outcome = (None, failure, 0.0)
                        else:
                            if completed.result is None:
                                raise RuntimeError("worker returned no candidate outcome")
                            outcome = completed.result
                        diagnostics[identifier] = ExecutionDiagnostic(
                            identifier,
                            "success"
                            if outcome[0] is not None
                            else completed.reason or "EstimatorFailure",
                            completed.elapsed,
                            outcome[1].error_type if outcome[1] is not None else None,
                        )
                        outcomes[identifier] = outcome
                        if workspace is not None and outcome[0] is not None:
                            workspace.save(outcome[0])
            except BaseException as exc:
                for identifier, diagnostic in diagnostics.items():
                    if diagnostic.outcome == "running":
                        diagnostics[identifier] = ExecutionDiagnostic(
                            identifier,
                            "Cancelled"
                            if isinstance(exc, KeyboardInterrupt)
                            else "CoordinatorError",
                            monotonic() - launched[identifier],
                            type(exc).__name__,
                        )
                raise
            finally:
                if workspace is not None:
                    _write_atomic(
                        workspace.path / "execution.json",
                        {
                            "schema_version": 1,
                            "policy": asdict(options),
                            "candidates": [
                                asdict(diagnostics[m.id or m.name]) for m in config.models
                            ],
                        },
                    )
        elif workers == 1:
            for model in pending:
                outcome = _execute_candidate(config, model, dataset, folds, model_registry, False)
                identifier = model.id or model.name
                outcomes[identifier] = outcome
                if workspace is not None and outcome[0] is not None:
                    workspace.save(outcome[0])
        elif pending:
            # Submit explicitly to commit each finished candidate as it arrives,
            # preserving already-durable successes if a different worker dies.
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                future_models = {
                    pool.submit(_candidate_process_entry, (config, model, dataset, folds)): model
                    for model in pending
                }
                for future in as_completed(future_models):
                    model = future_models[future]
                    outcome = future.result()
                    outcomes[model.id or model.name] = outcome
                    if workspace is not None and outcome[0] is not None:
                        workspace.save(outcome[0])

        candidates = [
            outcomes[model.id or model.name][0]
            for model in config.models
            if outcomes[model.id or model.name][0] is not None
        ]
        failures = [
            outcomes[model.id or model.name][1]
            for model in config.models
            if outcomes[model.id or model.name][1] is not None
        ]
        # The None branches were filtered above; explicit narrowing makes mypy
        # preserve the original public tuple types without casts.
        valid_candidates: list[CandidateResult] = [c for c in candidates if c is not None]
        valid_failures: list[CandidateFailure] = [f for f in failures if f is not None]

        primary_metric = config.primary_metric.value
        result = BenchmarkResult(
            fingerprint=config.fingerprint_for_dataset(dataset.identity),
            dataset=summarize_dataset(dataset),
            primary_metric=primary_metric,
            fold_count=len(folds),
            candidates=tuple(valid_candidates),
            failures=tuple(valid_failures),
            leaderboard=_build_leaderboard(
                valid_candidates,
                valid_failures,
                primary_metric,
                metric_spec(config.primary_metric).direction,
            ),
            total_duration_seconds=perf_counter() - started,
            execution_diagnostics=tuple(diagnostics[m.id or m.name] for m in config.models)
            if (options.bounded or workers > 1) and os.name == "posix"
            else (),
            execution_options=options
            if (options.bounded or workers > 1) and os.name == "posix"
            else None,
            checkpoint_workspace=checkpoint_dir,
            checkpoint_reused=tuple(workspace.reused) if workspace is not None else (),
            checkpoint_executed=tuple(workspace.executed) if workspace is not None else (),
            checkpoint_invalid=tuple(workspace.invalid) if workspace is not None else (),
        )
        if persist:
            directory = BenchmarkArtifactStore(config.output.directory).write(config, result)
            result = replace(result, artifact_directory=directory)
    return result
