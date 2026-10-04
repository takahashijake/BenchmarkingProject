"""Sequential orchestration over the existing leakage-safe fold kernel."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from benchforge.core.config import ModelConfig
from benchforge.core.seeds import derive_seed
from benchforge.data.registry import (
    Dataset,
    DatasetRegistry,
    default_dataset_registry,
    summarize_dataset,
)
from benchforge.evaluation.metrics import aggregate_fold_metrics, metric_spec
from benchforge.execution.runner import FoldExecutionError, PredictionRecord, execute_fold
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.robustness.config import RobustnessConfig
from benchforge.robustness.planner import RepetitionPlan, build_robustness_plan
from benchforge.robustness.results import (
    CandidateScore,
    Direction,
    MetricSummary,
    MetricValue,
    RankStability,
    RepetitionScores,
    RobustnessCandidateRepetitionResult,
    RobustnessFailure,
    RobustnessFoldResult,
    RobustnessLeaderboardEntry,
    RobustnessRepetitionResult,
    RobustnessResult,
)
from benchforge.robustness.statistics import (
    compare_all_pairs,
    rank_repetition,
    sample_std,
    summarize_ranks,
)


class RobustnessExecutionError(RuntimeError):
    """All candidates failed; evidence is available even when no ranking is possible."""

    def __init__(self, result: RobustnessResult) -> None:
        self.result = result
        artifact = str(result.artifact_directory) if result.artifact_directory else "not persisted"
        super().__init__(
            f"every robustness candidate failed in every repetition; artifacts: {artifact}"
        )


def _evaluate_candidate(
    config: RobustnessConfig,
    model: ModelConfig,
    dataset: Dataset,
    plan: RepetitionPlan,
    model_registry: ModelRegistry,
) -> RobustnessCandidateRepetitionResult:
    identifier = model.id or model.name
    estimator_seed = derive_seed(
        config.seed, "robustness", "estimator", plan.repetition_id, identifier
    )
    run_config = config.to_run_config(model).model_copy(update={"seed": estimator_seed})
    fold_results = []
    predictions: list[PredictionRecord] = []
    failure = None
    for position, fold in enumerate(plan.folds):
        try:
            result, records = execute_fold(run_config, dataset, fold, model_registry=model_registry)
            if not all(np.isfinite(value) for value in result.metrics.values()):
                raise ValueError("fold produced non-finite metrics")
        except (FoldExecutionError, ValueError) as exc:
            root = exc.__cause__ or exc
            failure = RobustnessFailure(
                plan.repetition_id,
                identifier,
                fold.fold_id,
                tuple(item.fold_id for item in plan.folds[position + 1 :]),
                type(root).__name__,
                str(root),
            )
            break
        fold_results.append(result)
        predictions.extend(records)
    metrics: tuple[MetricSummary, ...] = ()
    if failure is None:
        metrics = tuple(
            MetricSummary(name, value.mean, value.std)
            for name, value in sorted(
                aggregate_fold_metrics(item.metrics for item in fold_results).items()
            )
        )
    return RobustnessCandidateRepetitionResult(
        identifier,
        model.name,
        estimator_seed,
        "failed" if failure else "success",
        tuple(
            RobustnessFoldResult(
                item.fold_id,
                item.train_size,
                item.validation_size,
                tuple(MetricValue(name, value) for name, value in sorted(item.metrics.items())),
            )
            for item in fold_results
        ),
        metrics,
        tuple(sorted(predictions, key=lambda item: item.sample_index)),
        failure,
    )


def _leaderboard(
    config: RobustnessConfig,
    repetitions: tuple[RobustnessRepetitionResult, ...],
    ranks: tuple[RankStability, ...],
    direction: Direction,
) -> tuple[RobustnessLeaderboardEntry, ...]:
    entries = []
    for stability in ranks:
        candidates = tuple(
            candidate
            for repetition in repetitions
            for candidate in repetition.candidates
            if candidate.model_identifier == stability.model_identifier
            and candidate.status == "success"
        )
        scores = tuple(candidate.score(config.primary_metric.value) for candidate in candidates)
        completed = len(scores)
        failed = len(repetitions) - completed
        metrics = tuple(
            MetricSummary(metric.value, float(np.mean(values)), sample_std(values))
            for metric in sorted(config.metrics)
            for values in [tuple(candidate.score(metric.value) for candidate in candidates)]
            if values
        )
        entries.append(
            RobustnessLeaderboardEntry(
                None,
                stability.model_identifier,
                "failed" if not completed else "partial" if failed else "success",
                float(np.mean(scores)) if scores else None,
                sample_std(scores) if scores else None,
                min(scores) if scores else None,
                max(scores) if scores else None,
                stability.mean_rank,
                stability.rank_std,
                stability.first_place_rate,
                completed,
                failed,
                metrics,
            )
        )
    ordered = sorted(
        entries,
        key=lambda entry: (
            {"success": 0, "partial": 1, "failed": 2}[entry.status],
            (-entry.primary_metric_mean if direction == "maximize" else entry.primary_metric_mean)
            if entry.primary_metric_mean is not None
            else float("inf"),
            entry.model_identifier,
        ),
    )
    return tuple(
        replace(entry, rank=rank if entry.status != "failed" else None)
        for rank, entry in enumerate(ordered, 1)
    )


def run_robustness(
    config: RobustnessConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    persist: bool = True,
) -> RobustnessResult:
    started = perf_counter()
    dataset = dataset_registry.resolve(config.dataset)
    plan = build_robustness_plan(config, dataset, model_registry)
    direction = metric_spec(config.primary_metric).direction
    models = sorted(config.models, key=lambda model: model.id or model.name)
    repetitions = []
    observations = []
    for repetition in plan.repetitions:
        candidates = tuple(
            _evaluate_candidate(config, model, dataset, repetition, model_registry)
            for model in models
        )
        scores = RepetitionScores(
            repetition.repetition_id,
            tuple(
                CandidateScore(
                    candidate.model_identifier, candidate.score(config.primary_metric.value)
                )
                for candidate in candidates
                if candidate.status == "success"
            ),
        )
        observations.append(scores)
        repetitions.append(
            RobustnessRepetitionResult(
                repetition.repetition_id, candidates, rank_repetition(scores, direction)
            )
        )
    evidence = tuple(observations)
    ranks = summarize_ranks(plan.candidate_identifiers, evidence, direction)
    result = RobustnessResult(
        fingerprint=plan.fingerprint,
        dataset=summarize_dataset(dataset),
        primary_metric=config.primary_metric.value,
        optimization_direction=direction,
        plan=plan,
        repetitions=tuple(repetitions),
        pairwise_comparisons=compare_all_pairs(
            plan.candidate_identifiers,
            evidence,
            config.primary_metric.value,
            direction,
            config.robustness,
            config.seed,
        ),
        rank_stability=ranks,
        leaderboard=_leaderboard(config, tuple(repetitions), ranks, direction),
        failures=tuple(
            candidate.failure
            for repetition in repetitions
            for candidate in repetition.candidates
            if candidate.failure is not None
        ),
        total_duration_seconds=perf_counter() - started,
    )
    if persist:
        from benchforge.storage.robustness import RobustnessArtifactStore

        result = replace(
            result,
            artifact_directory=RobustnessArtifactStore(config.output.directory).write(
                config, result
            ),
        )
    if all(entry.status == "failed" for entry in result.leaderboard):
        raise RobustnessExecutionError(result)
    return result
