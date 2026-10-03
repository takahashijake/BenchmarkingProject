from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from benchforge.core.config import FileDatasetConfig, TaskType

if TYPE_CHECKING:
    from benchforge.data.registry import Dataset


def load_tabular_dataset(config: FileDatasetConfig) -> Dataset:
    path = config.path
    if not path.is_file():
        raise ValueError(f"dataset file does not exist: {path}")
    if config.source == "csv":
        _validate_csv_header(path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"could not read CSV dataset {path}: {exc}") from exc
    elif config.source == "parquet":
        _validate_parquet_columns(path)
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise ValueError(f"could not read Parquet dataset {path}: {exc}") from exc
    else:  # pragma: no cover
        raise ValueError(f"unsupported dataset source '{config.source}'")
    return dataset_from_frame(frame, config, _file_identity(path, config.source))


def dataset_from_frame(frame: pd.DataFrame, config: FileDatasetConfig, identity: str) -> Dataset:
    from benchforge.data.registry import Dataset

    columns = [str(column) for column in frame.columns]
    if len(columns) != len(set(columns)):
        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        raise ValueError(f"dataset contains duplicate column names: {', '.join(duplicates)}")
    if any(not isinstance(column, str) for column in frame.columns):
        frame = frame.copy()
        frame.columns = columns
    if config.target_column not in frame.columns:
        raise ValueError(f"target column '{config.target_column}' does not exist")
    missing_ignored = [name for name in config.id_columns if name not in frame.columns]
    if missing_ignored:
        raise ValueError(f"ignored/id columns do not exist: {', '.join(missing_ignored)}")
    if config.target_column in config.id_columns:
        raise ValueError("target column cannot also be an ignored/id column")
    if config.task != TaskType.BINARY_CLASSIFICATION:
        raise ValueError("V0.2 file ingestion supports only binary_classification")

    raw_target = frame[config.target_column]
    if raw_target.isna().any():
        raise ValueError(f"target column '{config.target_column}' contains missing values")
    unique_targets = list(pd.unique(raw_target))
    if len(unique_targets) != 2:
        raise ValueError(
            "binary-classification target must contain exactly 2 classes; "
            f"found {len(unique_targets)}"
        )
    ordered_targets = sorted(unique_targets, key=lambda value: (str(type(value)), str(value)))
    mapping = {value: index for index, value in enumerate(ordered_targets)}
    target = raw_target.map(mapping).astype(int).rename(config.target_column)

    features = frame.drop(columns=[config.target_column, *config.id_columns]).copy(deep=True)
    if features.shape[1] == 0:
        raise ValueError("dataset must contain at least one usable feature")

    numeric: list[str] = []
    categorical: list[str] = []
    for name in features.columns:
        series = features[name]
        if is_bool_dtype(series.dtype):
            categorical.append(name)
        elif is_datetime64_any_dtype(series.dtype):
            raise ValueError(f"feature '{name}' has unsupported datetime dtype")
        elif is_numeric_dtype(series.dtype):
            numeric.append(name)
        elif (
            is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            cardinality = int(series.nunique(dropna=True))
            if (
                cardinality > config.categorical_cardinality_limit
                and not config.allow_high_cardinality
            ):
                raise ValueError(
                    f"categorical feature '{name}' has cardinality {cardinality}, exceeding "
                    f"the limit {config.categorical_cardinality_limit}; exclude it, raise the "
                    "limit, or set allow_high_cardinality=true"
                )
            categorical.append(name)
        else:
            raise ValueError(f"feature '{name}' has unsupported dtype '{series.dtype}'")

    for name in categorical:
        features[name] = features[name].map(lambda value: str(value) if pd.notna(value) else value)

    feature_names = tuple(str(column) for column in features.columns)
    return Dataset(
        identity=identity,
        task=config.task,
        features=features.reset_index(drop=True),
        target=target.reset_index(drop=True),
        feature_names=feature_names,
        numeric_feature_names=tuple(numeric),
        categorical_feature_names=tuple(categorical),
        target_name=config.target_column,
        missing_values={name: int(features[name].isna().sum()) for name in feature_names},
        target_labels=(str(ordered_targets[0]), str(ordered_targets[1])),
    )


def _file_identity(path: Path, source: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{source}:sha256:{digest}"


def _validate_csv_header(path: Path) -> None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, UnicodeError, csv.Error) as exc:
        raise ValueError(f"could not read CSV header from {path}: {exc}") from exc
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"dataset contains duplicate column names: {', '.join(duplicates)}")


def _validate_parquet_columns(path: Path) -> None:
    try:
        import pyarrow.parquet as pq

        columns = pq.read_schema(path).names
    except Exception as exc:
        raise ValueError(f"could not read Parquet schema from {path}: {exc}") from exc
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"dataset contains duplicate column names: {', '.join(duplicates)}")
