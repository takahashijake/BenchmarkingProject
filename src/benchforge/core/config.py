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


class FileDatasetConfig(StrictModel):
    source: Literal["csv", "parquet"]
    path: Path
    target_column: str = Field(min_length=1)
    id_columns: tuple[str, ...] = ()
    task: TaskType
    categorical_cardinality_limit: int = Field(default=100, ge=2)
    allow_high_cardinality: bool = False

    @field_validator("id_columns")
    @classmethod
    def id_columns_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("id_columns must not contain duplicates")
        return value


DatasetSourceConfig = DatasetConfig | FileDatasetConfig


class SplitConfig(StrictModel):
    strategy: Literal["stratified_kfold"] = "stratified_kfold"
    n_splits: int = Field(default=5, ge=2)
    shuffle: Literal[True] = True


class PreprocessingConfig(StrictModel):
    median_imputation: bool = True
    standardize: bool = True


JsonScalar = str | int | float | bool | None


class ModelConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    id: str | None = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    parameters: dict[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def random_state_is_centralized(cls, value: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        if "random_state" in value:
            raise ValueError("model random_state is controlled by the top-level seed")
        return value

    @model_validator(mode="after")
    def identifiers_are_safe(self) -> ModelConfig:
        if self.name in {".", ".."} or self.id in {".", ".."}:
            raise ValueError("model name and id cannot be '.' or '..'")
        return self


class OutputConfig(StrictModel):
    directory: Path = Path("artifacts")


class RunConfig(StrictModel):
    schema_version: Literal[1] = 1
    task: TaskType
    dataset: DatasetSourceConfig
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
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the run task")
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
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


class BenchmarkConfig(StrictModel):
    schema_version: Literal[1] = 1
    task: TaskType
    dataset: DatasetSourceConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: tuple[ModelConfig, ...]
    metrics: tuple[MetricName, ...]
    primary_metric: MetricName
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("models")
    @classmethod
    def models_are_nonempty_and_distinct(
        cls, value: tuple[ModelConfig, ...]
    ) -> tuple[ModelConfig, ...]:
        if not value:
            raise ValueError("at least one model is required")
        canonical = [
            json.dumps(model.model_dump(mode="json", exclude_none=True), sort_keys=True)
            for model in value
        ]
        if len(set(canonical)) != len(canonical):
            raise ValueError("duplicate identical model configurations are not allowed")
        identifiers = [model.id or model.name for model in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("model identifiers must be unique; set an explicit model id")
        return value

    @field_validator("metrics")
    @classmethod
    def benchmark_metrics_are_nonempty_and_unique(
        cls, value: tuple[MetricName, ...]
    ) -> tuple[MetricName, ...]:
        if not value:
            raise ValueError("at least one metric is required")
        if len(set(value)) != len(value):
            raise ValueError("metrics must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_benchmark_semantics(self) -> BenchmarkConfig:
        if self.task != TaskType.BINARY_CLASSIFICATION:
            raise ValueError("V0.2 supports only task='binary_classification'")
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the benchmark task")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must also be present in metrics")
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        data["models"] = sorted(
            data["models"],
            key=lambda model: json.dumps(model, sort_keys=True, separators=(",", ":")),
        )
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

    def fingerprint_for_dataset(self, dataset_identity: str) -> str:
        benchmark_semantics = self.canonical_dict(include_output=False)
        dataset_semantics = benchmark_semantics["dataset"]
        if isinstance(dataset_semantics, dict) and "source" in dataset_semantics:
            dataset_semantics.pop("path", None)
        semantics = {
            "benchmark": benchmark_semantics,
            "resolved_dataset_identity": dataset_identity,
        }
        canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_run_config(self, model: ModelConfig) -> RunConfig:
        return RunConfig(
            task=self.task,
            dataset=self.dataset,
            split=self.split,
            preprocessing=self.preprocessing,
            model=model,
            metrics=self.metrics,
            seed=self.seed,
            output=self.output,
        )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return BenchmarkConfig.model_validate(raw)
