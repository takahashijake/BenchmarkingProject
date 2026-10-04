"""Normalized evidence contracts; missing historical metadata stays unavailable."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchforge.core.config import JsonScalar, SplitConfig


class ArtifactType(StrEnum):
    RUN = "run"
    BENCHMARK = "benchmark"
    SEARCH = "search"
    AUTOML = "automl"
    ROBUSTNESS = "robustness"


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Candidate(EvidenceModel):
    identifier: str
    model_name: str
    parameters: dict[str, JsonScalar]
    mode: str | None = None
    status: Literal["success", "partial", "failed"]
    rank: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class RepetitionSettings(EvidenceModel):
    repetitions: int
    confidence_level: float
    bootstrap_samples: int


class EvaluationEvidence(EvidenceModel):
    split: SplitConfig | None = None
    outer_split: SplitConfig | None = None
    inner_split: SplitConfig | None = None
    robustness: RepetitionSettings | None = None
    seed: int
    methodology_version: str | None = None
    scikit_learn_version: str | None = None
    fold_assignment_digest: str | None = None


class ArtifactDescriptor(EvidenceModel):
    artifact_type: ArtifactType
    directory: Path
    artifact_schema_version: int | None
    benchforge_version: str | None
    created_at: str | None
    fingerprint: str
    dataset_identity: str
    dataset_name: str | None
    task: str
    seed: int
    primary_metric: str | None
    optimization_direction: Literal["maximize", "minimize"] | None
    status: Literal["success", "partial", "failed"]
    failure_count: int
    candidates: tuple[Candidate, ...]
    evaluation: EvaluationEvidence
    summary_evidence: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any]
    source_metadata: dict[str, Any]


class Diagnostic(EvidenceModel):
    level: Literal["WARNING", "FAIL"]
    code: str
    message: str
    file: str | None = None


class VerificationReport(EvidenceModel):
    directory: Path
    descriptor: ArtifactDescriptor | None = None
    integrity: Literal["sealed", "legacy/unsealed", "invalid"]
    diagnostics: tuple[Diagnostic, ...] = ()
    deep: bool = False

    @property
    def passed(self) -> bool:
        return not any(item.level == "FAIL" for item in self.diagnostics)

    @property
    def outcome(self) -> str:
        return "PASS" if self.passed else "FAIL"


class ArtifactError(ValueError):
    """Invalid or unsupported persisted evidence, with location context."""
