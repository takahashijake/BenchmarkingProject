from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from benchforge.core.config import DatasetSourceConfig, FileDatasetConfig, TaskType
from benchforge.data.tabular import load_tabular_dataset


@dataclass(frozen=True)
class RegressionTargetSummary:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    unique_values: int


@dataclass(frozen=True)
class Dataset:
    identity: str
    task: TaskType
    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    target_name: str
    missing_values: dict[str, int]
    target_labels: tuple[str, ...] | None = None
    target_statistics: RegressionTargetSummary | None = None

    def __post_init__(self) -> None:
        if len(self.features) != len(self.target):
            raise ValueError("dataset features and target lengths differ")
        if tuple(self.features.columns) != self.feature_names:
            raise ValueError("dataset feature metadata does not match the feature frame")
        if set(self.numeric_feature_names) & set(self.categorical_feature_names):
            raise ValueError("numeric and categorical feature groups overlap")
        if set(self.numeric_feature_names) | set(self.categorical_feature_names) != set(
            self.feature_names
        ):
            raise ValueError("every feature must be classified as numeric or categorical")

    @property
    def row_count(self) -> int:
        return len(self.features)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class DatasetSummary:
    identity: str
    task: TaskType
    row_count: int
    feature_count: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_name: str
    target_labels: tuple[str, ...] | None
    target_statistics: RegressionTargetSummary | None
    missing_values: dict[str, int]


def summarize_dataset(dataset: Dataset) -> DatasetSummary:
    return DatasetSummary(
        identity=dataset.identity,
        task=dataset.task,
        row_count=dataset.row_count,
        feature_count=dataset.feature_count,
        numeric_features=dataset.numeric_feature_names,
        categorical_features=dataset.categorical_feature_names,
        target_name=dataset.target_name,
        target_labels=dataset.target_labels,
        target_statistics=dataset.target_statistics,
        missing_values=dict(dataset.missing_values),
    )


DatasetLoader = Callable[[], Dataset]


class DatasetRegistry:
    def __init__(self, loaders: dict[str, DatasetLoader]) -> None:
        self._loaders = MappingProxyType(dict(loaders))

    def resolve(self, config: str | DatasetSourceConfig) -> Dataset:
        if isinstance(config, FileDatasetConfig):
            return load_tabular_dataset(config)
        name = config if isinstance(config, str) else config.name
        try:
            dataset = self._loaders[name]()
        except KeyError as exc:
            supported = ", ".join(sorted(self._loaders))
            raise ValueError(f"unsupported dataset '{name}'; available: {supported}") from exc
        return _copy_dataset(dataset)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))


def _copy_dataset(dataset: Dataset) -> Dataset:
    return Dataset(
        identity=dataset.identity,
        task=dataset.task,
        features=dataset.features.copy(deep=True),
        target=dataset.target.copy(deep=True),
        feature_names=dataset.feature_names,
        numeric_feature_names=dataset.numeric_feature_names,
        categorical_feature_names=dataset.categorical_feature_names,
        target_name=dataset.target_name,
        missing_values=dict(dataset.missing_values),
        target_labels=dataset.target_labels,
        target_statistics=dataset.target_statistics,
    )


def _load_breast_cancer() -> Dataset:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame
    if frame is None:
        raise RuntimeError("scikit-learn did not return the requested DataFrame")
    target_name = str(bunch.target.name)
    features = frame.drop(columns=[target_name])
    target = frame[target_name].astype(int)
    feature_names = tuple(str(column) for column in features.columns)
    return Dataset(
        identity="sklearn:breast_cancer:v1",
        task=TaskType.BINARY_CLASSIFICATION,
        features=features,
        target=target,
        feature_names=feature_names,
        numeric_feature_names=feature_names,
        categorical_feature_names=(),
        target_name=target_name,
        missing_values={name: int(features[name].isna().sum()) for name in feature_names},
        target_labels=(str(bunch.target_names[0]), str(bunch.target_names[1])),
    )


def _load_diabetes() -> Dataset:
    bunch = load_diabetes(as_frame=True)
    frame = bunch.frame
    if frame is None:
        raise RuntimeError("scikit-learn did not return the requested DataFrame")
    target_name = str(bunch.target.name)
    features = frame.drop(columns=[target_name])
    target = frame[target_name].astype(float)
    feature_names = tuple(str(column) for column in features.columns)
    return Dataset(
        identity="sklearn:diabetes:v1",
        task=TaskType.REGRESSION,
        features=features,
        target=target,
        feature_names=feature_names,
        numeric_feature_names=feature_names,
        categorical_feature_names=(),
        target_name=target_name,
        missing_values={name: int(features[name].isna().sum()) for name in feature_names},
        target_statistics=RegressionTargetSummary(
            minimum=float(target.min()),
            maximum=float(target.max()),
            mean=float(target.mean()),
            standard_deviation=float(target.std(ddof=0)),
            unique_values=int(target.nunique()),
        ),
    )


def _load_iris() -> Dataset:
    bunch = load_iris(as_frame=True)
    frame = bunch.frame
    if frame is None:
        raise RuntimeError("scikit-learn did not return the requested DataFrame")
    target_name = str(bunch.target.name)
    features = frame.drop(columns=[target_name])
    target = frame[target_name].astype(int)
    feature_names = tuple(str(column) for column in features.columns)
    return Dataset(
        identity="sklearn:iris:v1",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        features=features,
        target=target,
        feature_names=feature_names,
        numeric_feature_names=feature_names,
        categorical_feature_names=(),
        target_name=target_name,
        missing_values={name: int(features[name].isna().sum()) for name in feature_names},
        target_labels=tuple(str(name) for name in bunch.target_names),
    )


default_dataset_registry = DatasetRegistry(
    {
        "breast_cancer": _load_breast_cancer,
        "diabetes": _load_diabetes,
        "iris": _load_iris,
    }
)
