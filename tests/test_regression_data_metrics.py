from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from benchforge.core.config import (
    FileDatasetConfig,
    MetricName,
    RunConfig,
    TaskType,
    load_run_config,
)
from benchforge.data import default_dataset_registry
from benchforge.evaluation.metrics import compute_metrics, metric_spec


def _file_config(path: Path, source: str = "csv") -> FileDatasetConfig:
    return FileDatasetConfig(
        source=source,
        path=path,
        target_column="target",
        task=TaskType.REGRESSION,
    )


def test_diabetes_is_continuous_regression_with_summary() -> None:
    dataset = default_dataset_registry.resolve("diabetes")
    assert dataset.identity == "sklearn:diabetes:v1"
    assert dataset.task == TaskType.REGRESSION
    assert dataset.features.shape == (442, 10)
    assert dataset.target.dtype.kind == "f"
    assert dataset.target.nunique() > 2
    assert dataset.target_labels is None
    assert dataset.target_statistics is not None
    assert dataset.target_statistics.unique_values == dataset.target.nunique()


@pytest.mark.parametrize("source", ["csv", "parquet"])
def test_regression_file_loads_and_integer_target_stays_regression(
    source: str, tmp_path: Path
) -> None:
    path = tmp_path / f"data.{source}"
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "kind": ["a", "b", "a"], "target": [10, 20, 35]})
    if source == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    dataset = default_dataset_registry.resolve(_file_config(path, source))
    assert dataset.task == TaskType.REGRESSION
    assert dataset.target.tolist() == [10.0, 20.0, 35.0]
    assert dataset.target.dtype.kind == "f"
    assert dataset.categorical_feature_names == ("kind",)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([1.0, None, 3.0], "contains missing values"),
        (["low", "medium", "high"], "must be numeric"),
        ([1.0, np.inf, 3.0], "only finite values"),
        ([4.0, 4.0, 4.0], "at least two distinct values"),
    ],
)
def test_invalid_regression_targets_fail(
    target: list[object], message: str, tmp_path: Path
) -> None:
    path = tmp_path / "invalid.parquet"
    pd.DataFrame({"x": [1, 2, 3], "target": target}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        default_dataset_registry.resolve(_file_config(path, "parquet"))


def test_empty_regression_target_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame(
        {
            "x": pd.Series(dtype=float),
            "target": pd.Series(dtype=float),
        }
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="at least one row"):
        default_dataset_registry.resolve(_file_config(path, "parquet"))


def test_regression_metrics_have_natural_values_and_directions() -> None:
    truth = np.asarray([1.0, 2.0, 4.0])
    prediction = np.asarray([2.0, 2.0, 2.0])
    values = compute_metrics(
        (
            MetricName.MEAN_ABSOLUTE_ERROR,
            MetricName.ROOT_MEAN_SQUARED_ERROR,
            MetricName.R2,
        ),
        truth,
        prediction,
        None,
        TaskType.REGRESSION,
    )
    assert values["mean_absolute_error"] == pytest.approx(1.0)
    assert values["root_mean_squared_error"] == pytest.approx(np.sqrt(5 / 3))
    assert values["r2"] == pytest.approx(-0.0714285714285714)
    assert metric_spec(MetricName.MEAN_ABSOLUTE_ERROR).direction == "minimize"
    assert metric_spec(MetricName.ROOT_MEAN_SQUARED_ERROR).direction == "minimize"
    assert metric_spec(MetricName.R2).direction == "maximize"


@pytest.mark.parametrize(
    ("config_path", "task", "strategy", "metrics", "message"),
    [
        (
            "configs/examples/diabetes_regression_run.yaml",
            "regression",
            "kfold",
            ["roc_auc"],
            "incompatible",
        ),
        (
            "configs/examples/breast_cancer_logreg.yaml",
            "binary_classification",
            "stratified_kfold",
            ["root_mean_squared_error"],
            "incompatible",
        ),
        (
            "configs/examples/diabetes_regression_run.yaml",
            "regression",
            "stratified_kfold",
            ["r2"],
            "split strategy",
        ),
        (
            "configs/examples/breast_cancer_logreg.yaml",
            "binary_classification",
            "kfold",
            ["accuracy"],
            "split strategy",
        ),
    ],
)
def test_task_metric_and_split_compatibility(
    config_path: str,
    task: str,
    strategy: str,
    metrics: list[str],
    message: str,
) -> None:
    data = load_run_config(config_path).model_dump(mode="json")
    data["task"] = task
    data["split"]["strategy"] = strategy
    data["metrics"] = metrics
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(data)
