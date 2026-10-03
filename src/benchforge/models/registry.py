from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from benchforge.core.config import ModelConfig

ModelFactory = Callable[[dict[str, Any], int], ClassifierMixin]


@dataclass(frozen=True)
class ModelCapabilities:
    requires_dense: bool = False


@dataclass(frozen=True)
class ModelSpec:
    factory: ModelFactory
    capabilities: ModelCapabilities = ModelCapabilities()


def _logistic_regression(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    return LogisticRegression(random_state=seed, **parameters)


def _random_forest(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    parameters.setdefault("n_jobs", 1)
    return RandomForestClassifier(random_state=seed, **parameters)


def _extra_trees(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    parameters.setdefault("n_jobs", 1)
    return ExtraTreesClassifier(random_state=seed, **parameters)


def _hist_gradient_boosting(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    return HistGradientBoostingClassifier(random_state=seed, **parameters)


def _dummy(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    parameters.setdefault("strategy", "prior")
    return DummyClassifier(random_state=seed, **parameters)


class ModelRegistry:
    def __init__(self, factories: dict[str, ModelFactory | ModelSpec]) -> None:
        specs = {
            name: value if isinstance(value, ModelSpec) else ModelSpec(value)
            for name, value in factories.items()
        }
        self._specs = MappingProxyType(specs)

    def create(self, config: ModelConfig, seed: int) -> ClassifierMixin:
        spec = self._resolve(config.name)
        try:
            return spec.factory(dict(config.parameters), seed)
        except TypeError as exc:
            raise ValueError(f"invalid parameters for model '{config.name}': {exc}") from exc

    def capabilities(self, name: str) -> ModelCapabilities:
        return self._resolve(name).capabilities

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
            _hist_gradient_boosting, ModelCapabilities(requires_dense=True)
        ),
        "dummy_classifier": ModelSpec(_dummy),
    }
)
