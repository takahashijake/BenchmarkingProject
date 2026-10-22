"""Immutable presentation evidence. JSON snapshots own polymorphic historical fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchforge.robustness.results import PairwiseComparison, RankStability


@dataclass(frozen=True)
class Evidence:
    """Canonical JSON and its artifact-relative source; decoding returns a private copy."""

    source: str
    json: str

    @classmethod
    def capture(cls, source: str, value: Any) -> Evidence:
        return cls(source, json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False))

    def decode(self) -> Any:
        return json.loads(self.json)


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    direction: Literal["maximize", "minimize"]
    primary: bool


@dataclass(frozen=True)
class Score:
    name: str
    mean: float
    std: float | None


@dataclass(frozen=True)
class TrialBudget:
    configured: int
    completed: int
    failed: int
    pruned: int


@dataclass(frozen=True)
class CandidateReport:
    identifier: str
    model_name: str
    mode: str | None
    status: str
    rank: int | None
    parameters: Evidence
    scores: tuple[Score, ...]
    evaluation_unit: str
    completed: int | None
    expected: int | None
    failures: int | None
    trials: TrialBudget | None


@dataclass(frozen=True)
class FoldReport:
    candidate: str
    fold_id: int
    repetition_id: int | None
    train_size: int
    validation_size: int
    metrics: tuple[tuple[str, float], ...]
    estimator_seed: int | None = None
    selection: Evidence | None = None


@dataclass(frozen=True)
class Verification:
    integrity: str
    deep: bool
    diagnostics: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ExperimentReport:
    directory: Path
    workflow: str
    fingerprint: str
    task: str
    dataset_identity: str
    dataset_name: str | None
    status: str
    primary_metric: str | None
    schema_version: int | None
    evaluator_version: str | None
    created_at: str | None
    verification: Verification
    metrics: tuple[Metric, ...]
    candidates: tuple[CandidateReport, ...]
    folds: tuple[FoldReport, ...]
    configuration: Evidence
    evaluation: Evidence
    provenance: Evidence
    dataset: Evidence | None
    failures: Evidence
    final_selection: Evidence | None
    plan: Evidence | None
    repetition_plan: Evidence | None
    repetitions: tuple[Evidence, ...]
    pairs: tuple[PairwiseComparison, ...]
    ranks: tuple[RankStability, ...]
    limitations: tuple[str, ...]
    decision: Evidence | None = None


@dataclass(frozen=True)
class CandidateChange:
    identifier: str
    score_a: float | None
    score_b: float | None
    delta_b_minus_a: float | None
    oriented_delta_b_minus_a: float | None
    parameters_changed: bool
    parameters_a: Evidence
    parameters_b: Evidence


@dataclass(frozen=True)
class ComparisonReport:
    a: ExperimentReport
    b: ExperimentReport
    compatibility: str
    reasons: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    changes: tuple[CandidateChange, ...]
    interpretation: str


class ReportingError(ValueError):
    """Invalid evidence, unsafe destination or failed report publication."""


class VerificationError(ReportingError):
    """Artifact verification failed; diagnostics retain their established codes."""
