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
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }


class MetricName(StrEnum):
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    ROC_AUC = "roc_auc"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    R2 = "r2"


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
    strategy: Literal["stratified_kfold", "kfold"] = "stratified_kfold"
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
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the run task")
        _validate_task_semantics(self.task, self.split, self.metrics)
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

    def fingerprint_for_dataset(self, dataset_identity: str) -> str:
        """Resolved persisted identity; config.fingerprint remains the unresolved API."""
        semantics = self.canonical_dict(include_output=False)
        semantics["metrics"] = sorted(semantics["metrics"])
        if "source" in semantics["dataset"]:
            semantics["dataset"].pop("path", None)
        payload = {
            "run": semantics,
            "resolved_dataset_identity": dataset_identity,
            "fingerprint_method": "resolved-run-v1",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


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
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the benchmark task")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must also be present in metrics")
        _validate_task_semantics(self.task, self.split, self.metrics)
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
        benchmark_semantics["metrics"] = sorted(benchmark_semantics["metrics"])
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


class SearchModelConfig(ModelConfig):
    mode: Literal["search", "fixed"] = "search"


class SearchBudget(StrictModel):
    trials_per_outer_fold: int = Field(ge=1)
    timeout_seconds_per_outer_fold: float | None = Field(default=None, gt=0)


class FinalSearchConfig(StrictModel):
    enabled: bool = False
    trials: int = Field(default=20, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def enabled_search_has_trials(self) -> FinalSearchConfig:
        if self.enabled and self.trials < 1:
            raise ValueError("enabled final search requires at least one trial")
        return self


class AutoMLBudget(StrictModel):
    total_trials: int = Field(gt=0)
    final_search_trials: int = Field(default=0, ge=0)
    timeout_seconds_per_outer_fold: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def final_reserve_leaves_an_outer_budget(self) -> AutoMLBudget:
        if self.final_search_trials >= self.total_trials:
            raise ValueError("final_search_trials must be less than total_trials")
        return self


class AutoMLModelPolicy(StrictModel):
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_fixed_baselines: bool = True

    @field_validator("include", "exclude")
    @classmethod
    def model_names_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("model policy entries must be unique")
        return value

    @model_validator(mode="after")
    def include_and_exclude_do_not_overlap(self) -> AutoMLModelPolicy:
        overlap = sorted(set(self.include) & set(self.exclude))
        if overlap:
            raise ValueError(f"include and exclude overlap: {', '.join(overlap)}")
        return self


class SearchConfig(StrictModel):
    schema_version: Literal[1] = 1
    task: TaskType
    dataset: DatasetSourceConfig
    outer_split: SplitConfig
    inner_split: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: tuple[SearchModelConfig, ...]
    metrics: tuple[MetricName, ...]
    primary_metric: MetricName
    budget: SearchBudget
    final_search: FinalSearchConfig = Field(default_factory=FinalSearchConfig)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("models")
    @classmethod
    def search_models_are_valid(
        cls, value: tuple[SearchModelConfig, ...]
    ) -> tuple[SearchModelConfig, ...]:
        if not value:
            raise ValueError("at least one search candidate is required")
        identifiers = [model.id or model.name for model in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("search candidate identifiers must be unique")
        if not any(model.mode == "search" for model in value):
            raise ValueError("at least one model must use mode='search'")
        return value

    @field_validator("metrics")
    @classmethod
    def search_metrics_are_nonempty_and_unique(
        cls, value: tuple[MetricName, ...]
    ) -> tuple[MetricName, ...]:
        if not value:
            raise ValueError("at least one metric is required")
        if len(set(value)) != len(value):
            raise ValueError("metrics must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_search_semantics(self) -> SearchConfig:
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the search task")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must also be present in metrics")
        _validate_task_semantics(self.task, self.outer_split, self.metrics)
        _validate_task_semantics(self.task, self.inner_split, self.metrics)
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

    def fingerprint_for_dataset(
        self, dataset_identity: str, search_space_identity: dict[str, Any] | str
    ) -> str:
        search_semantics = self.canonical_dict(include_output=False)
        search_semantics["metrics"] = sorted(search_semantics["metrics"])
        dataset_semantics = search_semantics["dataset"]
        if isinstance(dataset_semantics, dict) and "source" in dataset_semantics:
            dataset_semantics.pop("path", None)
        semantics = {
            "search": search_semantics,
            "resolved_dataset_identity": dataset_identity,
            "search_spaces": search_space_identity,
        }
        canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_run_config(
        self,
        model: ModelConfig,
        *,
        split: SplitConfig,
        seed: int,
    ) -> RunConfig:
        return RunConfig(
            task=self.task,
            dataset=self.dataset,
            split=split,
            preprocessing=self.preprocessing,
            model=model,
            metrics=self.metrics,
            seed=seed,
            output=self.output,
        )


def load_search_config(path: str | Path) -> SearchConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return SearchConfig.model_validate(raw)


class AutoMLConfig(StrictModel):
    schema_version: Literal[1] = 1
    task: TaskType
    dataset: DatasetSourceConfig
    outer_split: SplitConfig
    inner_split: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    metrics: tuple[MetricName, ...]
    primary_metric: MetricName
    budget: AutoMLBudget
    models: AutoMLModelPolicy = Field(default_factory=AutoMLModelPolicy)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("metrics")
    @classmethod
    def automl_metrics_are_nonempty_and_unique(
        cls, value: tuple[MetricName, ...]
    ) -> tuple[MetricName, ...]:
        if not value:
            raise ValueError("at least one metric is required")
        if len(set(value)) != len(value):
            raise ValueError("metrics must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_automl_semantics(self) -> AutoMLConfig:
        if isinstance(self.dataset, FileDatasetConfig) and self.dataset.task != self.task:
            raise ValueError("file dataset task must match the AutoML task")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must also be present in metrics")
        _validate_task_semantics(self.task, self.outer_split, self.metrics)
        _validate_task_semantics(self.task, self.inner_split, self.metrics)
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        data["models"]["include"] = sorted(data["models"]["include"])
        data["models"]["exclude"] = sorted(data["models"]["exclude"])
        if not include_output:
            data.pop("output", None)
        return data

    @property
    def fingerprint(self) -> str:
        """Stable unresolved policy identity; the output location is excluded."""
        canonical = json.dumps(
            self.canonical_dict(include_output=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def fingerprint_for_plan(
        self,
        *,
        dataset_identity: str,
        selected_models: dict[str, Any],
        search_spaces: dict[str, Any],
        allocation: dict[str, int],
        search_fingerprint: str,
    ) -> str:
        policy = self.canonical_dict(include_output=False)
        policy["metrics"] = sorted(policy["metrics"])
        dataset_semantics = policy["dataset"]
        if isinstance(dataset_semantics, dict) and "source" in dataset_semantics:
            dataset_semantics.pop("path", None)
        semantics = {
            "automl": policy,
            "resolved_dataset_identity": dataset_identity,
            "selected_models": selected_models,
            "search_spaces": search_spaces,
            "allocation": allocation,
            "generated_search_fingerprint": search_fingerprint,
        }
        canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_automl_config(path: str | Path) -> AutoMLConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return AutoMLConfig.model_validate(raw)


def _validate_task_semantics(
    task: TaskType, split: SplitConfig, metrics: tuple[MetricName, ...]
) -> None:
    expected_split = "stratified_kfold" if task.is_classification else "kfold"
    if split.strategy != expected_split:
        raise ValueError(
            f"split strategy '{split.strategy}' is incompatible with task '{task}'; "
            f"use '{expected_split}'"
        )
    # Local import keeps configuration contracts independent while making the metric
    # registry the single source of truth for compatibility.
    from benchforge.evaluation.metrics import metric_spec

    incompatible = [
        metric.value for metric in metrics if not metric_spec(metric).supports_task(task)
    ]
    if incompatible:
        raise ValueError(f"metrics {', '.join(incompatible)} are incompatible with task '{task}'")
