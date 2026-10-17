"""Deterministic content identity shared by datasets and checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from benchforge.data.registry import Dataset


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def dataset_digest(dataset: Dataset) -> str:
    """Hash resolved values, ordering, index, schema, and target, not a source path."""
    digest = hashlib.sha256()
    digest.update(
        _canonical(
            {
                "features": [
                    (str(name), str(dtype)) for name, dtype in dataset.features.dtypes.items()
                ],
                "target_dtype": str(dataset.target.dtype),
                "task": dataset.task.value,
                "numeric": dataset.numeric_feature_names,
                "categorical": dataset.categorical_feature_names,
            }
        )
    )
    digest.update(pd.util.hash_pandas_object(dataset.features, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(dataset.target, index=True).to_numpy().tobytes())
    return digest.hexdigest()
