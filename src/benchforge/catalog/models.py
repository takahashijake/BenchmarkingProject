"""Persistent index views; observations are scoped to physical copies."""

from pathlib import Path
from typing import Any, Literal

from benchforge.artifacts.types import ArtifactDescriptor, EvidenceModel, VerificationReport


class CatalogLocation(EvidenceModel):
    directory: Path
    descriptor: ArtifactDescriptor
    verification: VerificationReport
    evidence_digest: str


class CatalogExperiment(EvidenceModel):
    experiment_id: str
    artifact_type: str
    fingerprint: str
    task: str
    dataset_identity: str
    primary_metric: str | None
    locations: tuple[CatalogLocation, ...]


class CandidateDifference(EvidenceModel):
    identifier: str
    metric_a: float | None
    metric_b: float | None
    raw_delta_b_minus_a: float | None
    oriented_delta_b_minus_a: float | None
    rank_a: int | None
    rank_b: int | None
    parameters_a: dict[str, Any]
    parameters_b: dict[str, Any]
    parameters_changed: bool


class Comparison(EvidenceModel):
    experiment_a: str
    experiment_b: str
    compatibility: Literal["compatible", "partially compatible", "incompatible"]
    reasons: tuple[str, ...]
    fingerprint_changed: bool
    candidates_only_a: tuple[str, ...]
    candidates_only_b: tuple[str, ...]
    differences: tuple[CandidateDifference, ...]
    summary_evidence_a: dict[str, Any]
    summary_evidence_b: dict[str, Any]
    failure_count_a: int
    failure_count_b: int
    interpretation: str = (
        "Descriptive historical evidence only; metric differences do not establish "
        "statistical significance or superiority. Observations use the listed artifact locations."
    )
    location_a: Path
    location_b: Path
