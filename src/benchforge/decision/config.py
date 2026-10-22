"""Separate schema-1 policy for fixed, serial binary decision evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from benchforge.core.config import (
    DatasetSourceConfig,
    FileDatasetConfig,
    ModelConfig,
    OutputConfig,
    PreprocessingConfig,
    SplitConfig,
    StrictModel,
    TaskType,
)

Objective = Literal["f1", "balanced_accuracy", "misclassification_cost"]
PrimaryMetric = Literal[
    "selected.f1",
    "selected.balanced_accuracy",
    "selected.misclassification_cost",
    "selected.accuracy",
    "selected.precision",
    "selected.recall",
    "probability.brier_score",
    "probability.log_loss",
    "probability.roc_auc",
    "probability.ece",
]


class CalibrationConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal["none", "sigmoid", "isotonic"] = "none"
    n_splits: int = Field(default=3, ge=2, strict=True)
    isotonic_min_samples: int = Field(default=1000, ge=1000, strict=True)


class ThresholdConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    policy: Literal["fixed", "optimize"] = "fixed"
    value: float = Field(default=0.5, ge=0, le=1, strict=True)
    objective: Objective = "f1"
    candidates: tuple[Annotated[float, Field(strict=True)], ...] = tuple(i / 20 for i in range(21))
    false_positive_cost: float = Field(default=1.0, ge=0, strict=True)
    false_negative_cost: float = Field(default=1.0, ge=0, strict=True)

    @field_validator("candidates")
    @classmethod
    def bounded_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or len(value) > 10001:
            raise ValueError("threshold candidates need 1 to 10001 values")
        if any(not math.isfinite(t) or not 0 <= t <= 1 for t in value):
            raise ValueError("threshold candidates must be finite probabilities in [0, 1]")
        if len(set(value)) != len(value):
            raise ValueError("threshold candidates must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def fixed_value_is_not_ignored(self) -> ThresholdConfig:
        if self.policy == "optimize" and self.value != 0.5:
            raise ValueError("threshold.value applies only to fixed policy")
        return self


class DecisionConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    task: Literal[TaskType.BINARY_CLASSIFICATION] = TaskType.BINARY_CLASSIFICATION
    dataset: DatasetSourceConfig
    outer_split: SplitConfig = Field(default_factory=lambda: SplitConfig(n_splits=3))
    inner_split: SplitConfig = Field(default_factory=lambda: SplitConfig(n_splits=3))
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: tuple[ModelConfig, ...]
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    primary_metric: PrimaryMetric = "selected.f1"
    reliability_bins: int = Field(default=10, ge=2, le=100, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("schema_version", mode="before")
    @classmethod
    def explicit_version(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported decision configuration schema version")
        return value

    @model_validator(mode="after")
    def supported_policy(self) -> DecisionConfig:
        if not self.models:
            raise ValueError("at least one fixed candidate is required")
        ids = [m.id or m.name for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate identifiers must be unique")
        if any(s.strategy != "stratified_kfold" for s in (self.outer_split, self.inner_split)):
            raise ValueError("decision analysis requires stratified_kfold splits")
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must be binary_classification")
        for model in self.models:
            jobs = model.parameters.get("n_jobs", 1)
            if type(jobs) is not int or jobs != 1:
                raise ValueError("decision analysis requires estimator n_jobs=1")
            json.dumps(model.parameters, allow_nan=False)
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        data["models"] = sorted(data["models"], key=lambda m: m.get("id", m["name"]))
        if not include_output:
            data.pop("output", None)
            data["dataset"].pop("path", None)
        return data


def load_decision_config(path: str | Path) -> DecisionConfig:
    """Load a strict, independently versioned YAML decision policy."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load decision configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("decision configuration must contain a YAML mapping")
    return DecisionConfig.model_validate(raw)
