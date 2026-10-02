from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class MetricName(StrEnum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    ROC_AUC = "roc_auc"


class DatasetConfig(StrictModel):
    name: str = Field(min_length=1)


class SplitConfig(StrictModel):
    strategy: Literal["stratified_kfold"] = "stratified_kfold"
    n_splits: int = Field(default=5, ge=2)
    shuffle: Literal[True] = True


class PreprocessingConfig(StrictModel):
    median_imputation: bool = True
    standardize: bool = True


JsonScalar = str | int | float | bool | None


class ModelConfig(StrictModel):
    name: str = Field(min_length=1)
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def random_state_is_centralized(cls, value: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        if "random_state" in value:
            raise ValueError("model random_state is controlled by the top-level seed")
        return value


class OutputConfig(StrictModel):
    directory: Path = Path("artifacts")


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    task: TaskType
    dataset: DatasetConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig
    metrics: tuple[MetricName, ...]
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("metrics")
    @classmethod
    def metrics_are_nonempty_and_unique(
        cls, value: tuple[MetricName, ...]
    ) -> tuple[MetricName, ...]:
        if not value:
            raise ValueError("at least one metric is required")
        if len(set(value)) != len(value):
            raise ValueError("metrics must not contain duplicates")
        return value

    @model_validator(mode="after")
    def metrics_match_task(self) -> RunConfig:
        if self.task != TaskType.BINARY_CLASSIFICATION:
            raise ValueError("V0 supports only task='binary_classification'")
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if not include_output:
            data.pop("output", None)
        return data

    def canonical_json(self, *, include_output: bool = True) -> str:
        return json.dumps(
            self.canonical_dict(include_output=include_output),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def fingerprint(self) -> str:
        """Stable identity of experiment semantics; artifact location is excluded."""
        return hashlib.sha256(self.canonical_json(include_output=False).encode()).hexdigest()


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return RunConfig.model_validate(raw)
