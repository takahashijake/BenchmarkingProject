"""Validate plugin data before it enters the scientific kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchforge.core.config import TaskType
from benchforge.data.registry import Dataset


def validate_dataset(dataset: Dataset, tasks: tuple[str, ...]) -> None:
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset provider must return Dataset")
    if not isinstance(dataset.features, pd.DataFrame) or not isinstance(dataset.target, pd.Series):
        raise ValueError("provider dataset requires a DataFrame and a target Series")
    if not isinstance(dataset.task, TaskType) or not isinstance(dataset.identity, str):
        raise ValueError("provider dataset requires a typed task and string identity")
    dataset.__post_init__()
    if not dataset.identity or dataset.task.value not in tasks:
        raise ValueError("provider dataset identity/task disagrees with declaration")
    if not dataset.features.index.equals(pd.RangeIndex(len(dataset.features))):
        raise ValueError("provider dataset requires a consecutive RangeIndex")
    if not dataset.target.index.equals(dataset.features.index):
        raise ValueError("provider dataset target index must match features")
    if not dataset.features.columns.is_unique or not len(dataset.features.columns):
        raise ValueError("provider dataset requires unique, nonempty feature columns")
    if any(
        len(set(group)) != len(group)
        for group in (
            dataset.numeric_feature_names,
            dataset.categorical_feature_names,
        )
    ):
        raise ValueError("provider feature groups contain duplicates")
    if not len(dataset.target):
        raise ValueError("provider dataset is empty")
    values = dataset.target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("provider target must contain finite values")
    if dataset.task.is_classification:
        labels = np.unique(values)
        expected = 2 if dataset.task.value == "binary_classification" else len(labels)
        if (
            len(labels) != expected
            or expected < 2
            or not np.array_equal(labels, np.arange(expected))
            or dataset.target_labels is None
            or len(dataset.target_labels) != expected
        ):
            raise ValueError("classification targets require consecutive IDs and matching labels")
    for name in dataset.numeric_feature_names:
        if not pd.api.types.is_numeric_dtype(dataset.features[name]):
            raise ValueError(f"numeric feature '{name}' is not numeric")
        if np.isinf(dataset.features[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"numeric feature '{name}' contains infinity")
    missing = {name: int(dataset.features[name].isna().sum()) for name in dataset.feature_names}
    if dataset.missing_values != missing:
        raise ValueError("provider missing-value metadata disagrees with dataset")
