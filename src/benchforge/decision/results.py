"""Immutable scientific evidence: tuples, scalars and canonical JSON snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Index = Annotated[int, Field(strict=True, ge=0)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1)]
Binary = Annotated[int, Field(strict=True, ge=0, le=1)]
Number = Annotated[float, Field(strict=True)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Partition(Record):
    fold_id: Index
    train_indices: tuple[Index, ...]
    validation_indices: tuple[Index, ...]


class SplitPlan(Record):
    seed: Seed
    folds: tuple[Partition, ...]


class FitBoundary(Record):
    fold: Partition
    calibration: SplitPlan | None


class OuterPlan(Record):
    final: FitBoundary
    inner_seed: Seed
    inner: tuple[FitBoundary, ...]


class DecisionPlan(Record):
    outer_seed: Seed
    outer: tuple[OuterPlan, ...]


class DatasetEvidence(Record):
    identity: str
    content_digest: Digest
    target_digest: Digest
    row_count: Annotated[int, Field(strict=True, gt=0)]
    feature_count: Annotated[int, Field(strict=True, gt=0)]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_name: str
    target_labels: tuple[str, str]
    truth: tuple[Binary, ...]


class ProbabilityObservation(Record):
    sample_index: Index
    fold_id: Index
    truth: Binary
    probability: Probability


class DecisionPrediction(ProbabilityObservation):
    prediction: Binary
    reference_prediction: Binary


class Confusion(Record):
    tn: Index
    fp: Index
    fn: Index
    tp: Index


class MetricValue(Record):
    name: str
    value: Number


class PolicyEvaluation(Record):
    threshold: Probability
    confusion: Confusion
    metrics: tuple[MetricValue, ...]


class ReliabilityBin(Record):
    lower: Probability
    upper: Probability
    count: Index
    mean_probability: Probability | None
    positive_fraction: Probability | None


class ProbabilityQuality(Record):
    brier_score: Annotated[float, Field(ge=0, le=1, strict=True)]
    log_loss: Annotated[float, Field(ge=0, strict=True)]
    roc_auc: Probability | None
    ece: Probability
    bins: tuple[ReliabilityBin, ...]


class ThresholdTrial(Record):
    threshold: Probability
    confusion: Confusion
    objective_value: Number


class ThresholdSelection(Record):
    threshold: Probability
    objective_value: Number
    trials: tuple[ThresholdTrial, ...]


class InnerEvidence(Record):
    fold_id: Index
    estimator_seed: Seed
    probabilities: tuple[ProbabilityObservation, ...]


class OuterEvaluation(Record):
    fold_id: Index
    estimator_seed: Seed
    inner: tuple[InnerEvidence, ...]
    selection: ThresholdSelection
    predictions: tuple[DecisionPrediction, ...]
    selected: PolicyEvaluation
    reference: PolicyEvaluation
    probability_quality: ProbabilityQuality


class Aggregate(Record):
    name: str
    mean: Number
    std: Annotated[float, Field(ge=0, strict=True)]


class Failure(Record):
    fold_id: Index
    error_type: str
    message: str


class CandidateResult(Record):
    identifier: str
    model_name: str
    parameters_json: str
    status: Literal["success", "partial", "failed"]
    folds: tuple[OuterEvaluation, ...]
    failure: Failure | None
    metrics: tuple[Aggregate, ...]
    rank: Annotated[int, Field(strict=True, gt=0)] | None = None


class DecisionResult(Record):
    artifact_schema_version: Literal[1]
    fingerprint: Digest
    identity_json: str
    dataset: DatasetEvidence
    plan: DecisionPlan
    primary_metric: str
    optimization_direction: Literal["maximize", "minimize"]
    status: Literal["success", "partial", "failed"]
    candidates: tuple[CandidateResult, ...]
    artifact_directory: Path | None = Field(default=None, exclude=True)
