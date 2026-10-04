"""Strict fixed-candidate robustness configuration, sharing benchmark semantics."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from benchforge.core.config import BenchmarkConfig, DatasetConfig, StrictModel


class RobustnessSettings(StrictModel):
    repetitions: int = Field(default=5, ge=2, strict=True)
    confidence_level: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)
    bootstrap_samples: int = Field(default=2000, ge=100, le=100_000, strict=True)


class RobustnessConfig(BenchmarkConfig):
    robustness: RobustnessSettings = Field(default_factory=RobustnessSettings)

    @model_validator(mode="after")
    def validate_robustness_semantics(self) -> "RobustnessConfig":
        from benchforge.data.registry import default_dataset_registry
        from benchforge.models.registry import default_model_registry

        if len(self.models) < 2:
            raise ValueError("robustness requires at least two candidate models")
        for model in self.models:
            if not default_model_registry.supports_task(model.name, self.task):
                raise ValueError(f"model '{model.name}' is incompatible with task '{self.task}'")
        json.dumps(self.canonical_dict(include_output=False), allow_nan=False)
        if isinstance(self.dataset, DatasetConfig):
            dataset = default_dataset_registry.resolve(self.dataset)
            if dataset.task != self.task:
                raise ValueError("dataset task must match the robustness task")
        return self

    def canonical_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        data = super().canonical_dict(include_output=include_output)
        for model in data["models"]:
            model["id"] = model.get("id", model["name"])
        data["models"] = sorted(data["models"], key=lambda model: model.get("id", model["name"]))
        data["metrics"] = sorted(data["metrics"])
        return data

    def fingerprint_for_dataset(self, dataset_identity: str) -> str:
        from benchforge.robustness.planner import robustness_fingerprint

        return robustness_fingerprint(self, dataset_identity)


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return RobustnessConfig.model_validate(raw)
