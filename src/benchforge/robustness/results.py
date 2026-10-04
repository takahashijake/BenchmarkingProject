"""Immutable robustness evidence; natural scores and oriented effects remain distinct."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchforge.data.registry import DatasetSummary
from benchforge.execution.runner import PredictionRecord
from benchforge.robustness.planner import RobustnessPlan

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float


@dataclass(frozen=True)
class MetricSummary:
    name: str
    mean: float
    std: float


@dataclass(frozen=True)
class RobustnessFoldResult:
    fold_id: int
    train_size: int
    validation_size: int
    metrics: tuple[MetricValue, ...]


@dataclass(frozen=True)
class RobustnessFailure:
    repetition_id: int
    model_identifier: str
    fold_id: int
    skipped_fold_ids: tuple[int, ...]
    error_type: str
    message: str


@dataclass(frozen=True)
class RobustnessCandidateRepetitionResult:
    model_identifier: str
    model_name: str
    estimator_seed: int
    status: Literal["success", "failed"]
    fold_results: tuple[RobustnessFoldResult, ...]
    metrics: tuple[MetricSummary, ...]
    predictions: tuple[PredictionRecord, ...]
    failure: RobustnessFailure | None

    def score(self, metric: str) -> float:
        for item in self.metrics:
            if item.name == metric:
                return item.mean
        raise ValueError(f"metric {metric!r} is unavailable for {self.model_identifier!r}")


@dataclass(frozen=True)
class CandidateScore:
    model_identifier: str
    value: float


@dataclass(frozen=True)
class RepetitionScores:
    repetition_id: int
    scores: tuple[CandidateScore, ...]


@dataclass(frozen=True)
class CandidateRank:
    model_identifier: str
    rank: int


@dataclass(frozen=True)
class RobustnessRepetitionResult:
    repetition_id: int
    candidates: tuple[RobustnessCandidateRepetitionResult, ...]
    ranks: tuple[CandidateRank, ...]


@dataclass(frozen=True)
class ConfidenceInterval:
    confidence_level: float
    lower_bound: float | None
    upper_bound: float | None
    bootstrap_samples: int
    seed: int
    method: str = "paired-repetition percentile bootstrap of mean (PCG64, linear quantiles)"


@dataclass(frozen=True)
class PairwiseComparison:
    candidate_a: str
    candidate_b: str
    primary_metric: str
    optimization_direction: Direction
    usable_repetitions: int
    repetition_ids: tuple[int, ...]
    raw_paired_deltas: tuple[float, ...]
    paired_deltas: tuple[float, ...]
    mean_delta: float | None
    median_delta: float | None
    std_delta: float | None
    minimum_delta: float | None
    maximum_delta: float | None
    a_wins: int
    b_wins: int
    ties: int
    a_win_rate: float | None
    b_win_rate: float | None
    interval: ConfidenceInterval
    interpretation: str = "positive delta means candidate A performed better than candidate B"


@dataclass(frozen=True)
class RankFrequency:
    rank: int
    count: int


@dataclass(frozen=True)
class RankStability:
    model_identifier: str
    ranked_repetitions: int
    missing_repetitions: int
    mean_rank: float | None
    median_rank: float | None
    rank_std: float | None
    best_rank: int | None
    worst_rank: int | None
    first_place_count: int
    first_place_rate: float
    top_2_count: int
    top_2_rate: float
    rank_counts: tuple[RankFrequency, ...]


@dataclass(frozen=True)
class RobustnessLeaderboardEntry:
    rank: int | None
    model_identifier: str
    status: Literal["success", "partial", "failed"]
    primary_metric_mean: float | None
    primary_metric_std: float | None
    minimum_score: float | None
    maximum_score: float | None
    mean_rank: float | None
    rank_std: float | None
    first_place_rate: float
    completed_repetitions: int
    failed_repetitions: int
    metrics: tuple[MetricSummary, ...]


@dataclass(frozen=True)
class RobustnessResult:
    fingerprint: str
    dataset: DatasetSummary
    primary_metric: str
    optimization_direction: Direction
    plan: RobustnessPlan
    repetitions: tuple[RobustnessRepetitionResult, ...]
    pairwise_comparisons: tuple[PairwiseComparison, ...]
    rank_stability: tuple[RankStability, ...]
    leaderboard: tuple[RobustnessLeaderboardEntry, ...]
    failures: tuple[RobustnessFailure, ...]
    total_duration_seconds: float
    artifact_directory: Path | None = None
