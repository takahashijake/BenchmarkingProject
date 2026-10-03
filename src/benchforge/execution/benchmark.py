from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from benchforge.core.config import BenchmarkConfig, ModelConfig
from benchforge.data.registry import (
    Dataset,
    DatasetRegistry,
    DatasetSummary,
    default_dataset_registry,
)
from benchforge.evaluation.metrics import AggregateMetric
from benchforge.execution.runner import RunResult, run_benchmark
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.splits.stratified import build_stratified_folds
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


def summarize_dataset(dataset: Dataset) -> DatasetSummary:
    return DatasetSummary(
        identity=dataset.identity,
        row_count=dataset.row_count,
        feature_count=dataset.feature_count,
        numeric_features=dataset.numeric_feature_names,
        categorical_features=dataset.categorical_feature_names,
        target_name=dataset.target_name,
        target_labels=dataset.target_labels,
        missing_values=dict(dataset.missing_values),
    )


def _build_leaderboard(
    candidates: list[CandidateResult],
    failures: list[CandidateFailure],
    primary_metric: str,
) -> tuple[LeaderboardEntry, ...]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.run_result.aggregate_metrics[primary_metric].mean,
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


def run_benchmark_suite(
    config: BenchmarkConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    persist: bool = True,
) -> BenchmarkResult:
    started = perf_counter()
    dataset = dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError(
            f"dataset task '{dataset.task}' does not match configured task '{config.task}'"
        )
    folds = build_stratified_folds(dataset.target, config.split, config.seed)
    candidates: list[CandidateResult] = []
    failures: list[CandidateFailure] = []

    for model in config.models:
        identifier = model.id or model.name
        candidate_started = perf_counter()
        try:
            run_result = run_benchmark(
                config.to_run_config(model),
                dataset_registry=dataset_registry,
                model_registry=model_registry,
                persist=False,
                dataset=dataset,
                folds=folds,
            )
        except Exception as exc:
            failures.append(
                CandidateFailure(
                    model_identifier=identifier,
                    model_name=model.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    duration_seconds=perf_counter() - candidate_started,
                )
            )
            continue
        candidates.append(
            CandidateResult(
                model_identifier=identifier,
                model_config=model,
                run_result=run_result,
                duration_seconds=perf_counter() - candidate_started,
            )
        )

    primary_metric = config.primary_metric.value
    result = BenchmarkResult(
        fingerprint=config.fingerprint_for_dataset(dataset.identity),
        dataset=summarize_dataset(dataset),
        primary_metric=primary_metric,
        fold_count=len(folds),
        candidates=tuple(candidates),
        failures=tuple(failures),
        leaderboard=_build_leaderboard(candidates, failures, primary_metric),
        total_duration_seconds=perf_counter() - started,
    )
    if persist:
        directory = BenchmarkArtifactStore(config.output.directory).write(config, result)
        result = replace(result, artifact_directory=directory)
    return result
