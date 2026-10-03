from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from benchforge.core.config import ModelConfig, TaskType

ModelFactory = Callable[[dict[str, Any], int], BaseEstimator]


@dataclass(frozen=True)
class ModelCapabilities:
    requires_dense: bool = False


@dataclass(frozen=True)
class ModelSpec:
    factory: ModelFactory
    task: TaskType = TaskType.BINARY_CLASSIFICATION
    capabilities: ModelCapabilities = ModelCapabilities()


def _logistic_regression(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    # sklearn 1.8 replaced the deprecated penalty selector with l1_ratio. Keep the
    # BenchForge model contract stable across supported sklearn releases.
    version = tuple(int(part) for part in sklearn_version.split(".")[:2])
    penalty = parameters.get("penalty")
    if version >= (1, 8) and penalty in {"l1", "l2"}:
        parameters.pop("penalty")
        parameters.setdefault("l1_ratio", 1.0 if penalty == "l1" else 0.0)
    return LogisticRegression(random_state=seed, **parameters)


def _random_forest(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    parameters.setdefault("n_jobs", 1)
    return RandomForestClassifier(random_state=seed, **parameters)


def _extra_trees(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    parameters.setdefault("n_jobs", 1)
    return ExtraTreesClassifier(random_state=seed, **parameters)


def _hist_gradient_boosting(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(random_state=seed, **parameters)


def _dummy(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    parameters.setdefault("strategy", "prior")
    return DummyClassifier(random_state=seed, **parameters)


def _linear_regression(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    del seed
    return LinearRegression(**parameters)


def _ridge_regressor(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    del seed
    return Ridge(**parameters)


def _random_forest_regressor(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    parameters.setdefault("n_jobs", 1)
    return RandomForestRegressor(random_state=seed, **parameters)


def _extra_trees_regressor(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    parameters.setdefault("n_jobs", 1)
    return ExtraTreesRegressor(random_state=seed, **parameters)


def _hist_gradient_boosting_regressor(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    return HistGradientBoostingRegressor(random_state=seed, **parameters)


def _dummy_regressor(parameters: dict[str, Any], seed: int) -> BaseEstimator:
    del seed
    parameters.setdefault("strategy", "mean")
    return DummyRegressor(**parameters)


class ModelRegistry:
    def __init__(self, factories: dict[str, ModelFactory | ModelSpec]) -> None:
        specs = {
            name: value if isinstance(value, ModelSpec) else ModelSpec(value)
            for name, value in factories.items()
        }
        self._specs = MappingProxyType(specs)

    def create(self, config: ModelConfig, seed: int, task: TaskType | None = None) -> BaseEstimator:
        spec = self._resolve(config.name)
        if task is not None and spec.task != task:
            raise ValueError(
                f"model '{config.name}' supports task '{spec.task}', not configured task '{task}'"
            )
        try:
            return spec.factory(dict(config.parameters), seed)
        except TypeError as exc:
            raise ValueError(f"invalid parameters for model '{config.name}': {exc}") from exc

    def capabilities(self, name: str) -> ModelCapabilities:
        return self._resolve(name).capabilities

    def task(self, name: str) -> TaskType:
        return self._resolve(name).task

    def _resolve(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(f"unsupported model '{name}'; available: {supported}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


default_model_registry = ModelRegistry(
    {
        "logistic_regression": ModelSpec(_logistic_regression),
        "random_forest_classifier": ModelSpec(_random_forest),
        "extra_trees_classifier": ModelSpec(_extra_trees),
        "hist_gradient_boosting_classifier": ModelSpec(
            _hist_gradient_boosting,
            capabilities=ModelCapabilities(requires_dense=True),
        ),
        "dummy_classifier": ModelSpec(_dummy),
        "linear_regression": ModelSpec(_linear_regression, TaskType.REGRESSION),
        "ridge_regressor": ModelSpec(_ridge_regressor, TaskType.REGRESSION),
        "random_forest_regressor": ModelSpec(_random_forest_regressor, TaskType.REGRESSION),
        "extra_trees_regressor": ModelSpec(_extra_trees_regressor, TaskType.REGRESSION),
        "hist_gradient_boosting_regressor": ModelSpec(
            _hist_gradient_boosting_regressor,
            TaskType.REGRESSION,
            ModelCapabilities(requires_dense=True),
        ),
        "dummy_regressor": ModelSpec(_dummy_regressor, TaskType.REGRESSION),
    }
)
