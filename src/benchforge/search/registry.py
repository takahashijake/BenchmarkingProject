from __future__ import annotations

from types import MappingProxyType
from typing import Any

from benchforge.search.spaces import (
    EXTRA_TREES_REGRESSOR_SPACE,
    EXTRA_TREES_SPACE,
    HIST_GRADIENT_BOOSTING_REGRESSOR_SPACE,
    HIST_GRADIENT_BOOSTING_SPACE,
    LOGISTIC_SPACE,
    RANDOM_FOREST_REGRESSOR_SPACE,
    RANDOM_FOREST_SPACE,
    RIDGE_REGRESSOR_SPACE,
    SearchSpace,
)


class SearchSpaceRegistry:
    def __init__(self, spaces: dict[str, SearchSpace]) -> None:
        self._spaces = MappingProxyType(dict(spaces))

    def resolve(self, model_name: str) -> SearchSpace:
        try:
            return self._spaces[model_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._spaces))
            raise ValueError(
                f"unsupported search family '{model_name}'; searchable families: {supported}"
            ) from exc

    def identity_for(self, model_names: set[str]) -> dict[str, Any]:
        return {name: self.resolve(name).identity for name in sorted(model_names)}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._spaces))


default_search_space_registry = SearchSpaceRegistry(
    {
        "logistic_regression": LOGISTIC_SPACE,
        "random_forest_classifier": RANDOM_FOREST_SPACE,
        "extra_trees_classifier": EXTRA_TREES_SPACE,
        "hist_gradient_boosting_classifier": HIST_GRADIENT_BOOSTING_SPACE,
        "ridge_regressor": RIDGE_REGRESSOR_SPACE,
        "random_forest_regressor": RANDOM_FOREST_REGRESSOR_SPACE,
        "extra_trees_regressor": EXTRA_TREES_REGRESSOR_SPACE,
        "hist_gradient_boosting_regressor": HIST_GRADIENT_BOOSTING_REGRESSOR_SPACE,
    }
)
