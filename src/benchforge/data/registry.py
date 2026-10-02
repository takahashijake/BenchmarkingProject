from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd
from sklearn.datasets import load_breast_cancer

from benchforge.core.config import TaskType


@dataclass(frozen=True)
class Dataset:
    identity: str
    task: TaskType
    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.target):
            raise ValueError("dataset features and target lengths differ")


DatasetLoader = Callable[[], Dataset]


class DatasetRegistry:
    def __init__(self, loaders: dict[str, DatasetLoader]) -> None:
        self._loaders = MappingProxyType(dict(loaders))

    def resolve(self, name: str) -> Dataset:
        try:
            dataset = self._loaders[name]()
        except KeyError as exc:
            supported = ", ".join(sorted(self._loaders))
            raise ValueError(f"unsupported dataset '{name}'; available: {supported}") from exc
        # Isolate each run from loader or caller mutation.
        return Dataset(
            identity=dataset.identity,
            task=dataset.task,
            features=dataset.features.copy(deep=True),
            target=dataset.target.copy(deep=True),
            feature_names=dataset.feature_names,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))


def _load_breast_cancer() -> Dataset:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame
    if frame is None:
        raise RuntimeError("scikit-learn did not return the requested DataFrame")
    target_name = str(bunch.target.name)
    features = frame.drop(columns=[target_name])
    target = frame[target_name].astype(int)
    return Dataset(
        identity="sklearn:breast_cancer:v1",
        task=TaskType.BINARY_CLASSIFICATION,
        features=features,
        target=target,
        feature_names=tuple(str(column) for column in features.columns),
    )


default_dataset_registry = DatasetRegistry({"breast_cancer": _load_breast_cancer})
