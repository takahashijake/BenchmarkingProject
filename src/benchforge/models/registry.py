from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from benchforge.core.config import ModelConfig

ModelFactory = Callable[[dict[str, Any], int], ClassifierMixin]


def _logistic_regression(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    return LogisticRegression(random_state=seed, **parameters)


def _random_forest(parameters: dict[str, Any], seed: int) -> ClassifierMixin:
    return RandomForestClassifier(random_state=seed, **parameters)


class ModelRegistry:
    def __init__(self, factories: dict[str, ModelFactory]) -> None:
        self._factories = MappingProxyType(dict(factories))

    def create(self, config: ModelConfig, seed: int) -> ClassifierMixin:
        try:
            factory = self._factories[config.name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(f"unsupported model '{config.name}'; available: {supported}") from exc
        try:
            return factory(dict(config.parameters), seed)
        except TypeError as exc:
            raise ValueError(f"invalid parameters for model '{config.name}': {exc}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


default_model_registry = ModelRegistry(
    {
        "logistic_regression": _logistic_regression,
        "random_forest_classifier": _random_forest,
    }
)
