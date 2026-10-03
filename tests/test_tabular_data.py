from pathlib import Path

import pandas as pd
import pytest

from benchforge.core.config import FileDatasetConfig, TaskType
from benchforge.data import default_dataset_registry


def _config(path: Path, source: str = "csv", **changes: object) -> FileDatasetConfig:
    values: dict[str, object] = {
        "source": source,
        "path": path,
        "target_column": "target",
        "id_columns": ["row_id"],
        "task": "binary_classification",
    }
    values.update(changes)
    return FileDatasetConfig.model_validate(values)


def test_csv_dataset_loads_and_infers_schema() -> None:
    config = FileDatasetConfig(
        source="csv",
        path=Path("configs/examples/data/mixed_customers.csv"),
        target_column="subscribed",
        id_columns=("customer_id",),
        task=TaskType.BINARY_CLASSIFICATION,
    )
    dataset = default_dataset_registry.resolve(config)
    assert dataset.row_count == 30
    assert dataset.feature_count == 5
    assert "subscribed" not in dataset.features
    assert "customer_id" not in dataset.features
    assert dataset.numeric_feature_names == ("age", "income", "visits")
    assert dataset.categorical_feature_names == ("segment", "uses_mobile")
    assert dataset.missing_values["income"] == 3
    assert dataset.missing_values["segment"] == 2
    assert set(dataset.target) == {0, 1}


def test_parquet_dataset_loads(tmp_path: Path) -> None:
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "amount": [1.0, None, 3.0, 4.0],
            "kind": ["a", "b", None, "a"],
            "target": ["no", "yes", "no", "yes"],
        }
    ).to_parquet(path, index=False)
    dataset = default_dataset_registry.resolve(_config(path, source="parquet"))
    assert dataset.row_count == 4
    assert dataset.numeric_feature_names == ("amount",)
    assert dataset.categorical_feature_names == ("kind",)
    assert dataset.identity.startswith("parquet:sha256:")


def test_invalid_target_and_non_binary_target_fail(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"row_id": [1, 2, 3], "value": [1, 2, 3], "target": [0, 1, 2]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="exactly 2 classes"):
        default_dataset_registry.resolve(_config(path))
    with pytest.raises(ValueError, match="target column 'missing' does not exist"):
        default_dataset_registry.resolve(_config(path, target_column="missing"))


def test_missing_id_column_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"value": [1, 2], "target": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ignored/id columns do not exist: row_id"):
        default_dataset_registry.resolve(_config(path))


def test_high_cardinality_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "row_id": range(6),
            "free_text": [f"unique value {index}" for index in range(6)],
            "target": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cardinality 6, exceeding the limit 3"):
        default_dataset_registry.resolve(_config(path, categorical_cardinality_limit=3))
    dataset = default_dataset_registry.resolve(
        _config(path, categorical_cardinality_limit=3, allow_high_cardinality=True)
    )
    assert dataset.categorical_feature_names == ("free_text",)
