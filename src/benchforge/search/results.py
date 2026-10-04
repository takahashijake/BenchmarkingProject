from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchforge.core.config import JsonScalar, SearchModelConfig
from benchforge.data.registry import DatasetSummary
from benchforge.evaluation.metrics import AggregateMetric
from benchforge.execution.runner import FoldResult, PredictionRecord
from benchforge.splits.stratified import Fold


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    proposed_parameters: dict[str, JsonScalar]
    parameters: dict[str, JsonScalar]
    value: float | None
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class OuterFoldSearchResult:
    fold_id: int
    outer_train_indices: tuple[int, ...]
    outer_validation_indices: tuple[int, ...]
    inner_folds: tuple[Fold, ...]
    selected_parameters: dict[str, JsonScalar]
    best_inner_score: float | None
    trials: tuple[TrialRecord, ...]
    outer_result: FoldResult
    outer_predictions: tuple[PredictionRecord, ...]
    duration_seconds: float
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    sampler_seed: int | None


@dataclass(frozen=True)
class FamilySearchResult:
    model_identifier: str
    candidate: SearchModelConfig
    outer_folds: tuple[OuterFoldSearchResult, ...]
    aggregate_metrics: dict[str, AggregateMetric]
    predictions: tuple[PredictionRecord, ...]
    duration_seconds: float
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int


@dataclass(frozen=True)
class FamilySearchFailure:
    model_identifier: str
    model_name: str
    error_type: str
    message: str
    duration_seconds: float
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    trials: tuple[TrialRecord, ...]


@dataclass(frozen=True)
class TunedLeaderboardEntry:
    rank: int | None
    model_identifier: str
    mode: str
    status: str
    primary_metric_mean: float | None
    primary_metric_std: float | None
    metrics: dict[str, AggregateMetric]
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    duration_seconds: float


@dataclass(frozen=True)
class FinalCandidate:
    model_identifier: str
    model_name: str
    mode: str
    parameters: dict[str, JsonScalar]
    cv_selection_score: float
    trials: tuple[TrialRecord, ...]
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    duration_seconds: float


@dataclass(frozen=True)
class SearchResult:
    fingerprint: str
    dataset: DatasetSummary
    primary_metric: str
    optimization_direction: str
    outer_fold_count: int
    inner_fold_count: int
    outer_folds: tuple[Fold, ...]
    families: tuple[FamilySearchResult, ...]
    failures: tuple[FamilySearchFailure, ...]
    leaderboard: tuple[TunedLeaderboardEntry, ...]
    final_candidate: FinalCandidate | None
    total_duration_seconds: float
    artifact_directory: Path | None = None
    search_space_identity: dict[str, Any] = field(default_factory=dict)
