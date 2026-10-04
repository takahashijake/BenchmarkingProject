from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from benchforge.core.config import (
    FileDatasetConfig,
    MetricName,
    ModelConfig,
    RunConfig,
    SplitConfig,
    TaskType,
    load_run_config,
)
from benchforge.data import default_dataset_registry
from benchforge.data.tabular import dataset_from_frame
from benchforge.evaluation.metrics import compute_metrics, metric_spec
from benchforge.models import default_model_registry
from benchforge.splits import build_folds, build_inner_folds


def _file_config(path: Path, task: TaskType) -> FileDatasetConfig:
    return FileDatasetConfig(
        source="csv",
        path=path,
        target_column="target",
        task=task,
    )


def test_iris_registry_metadata_and_copy_isolation() -> None:
    first = default_dataset_registry.resolve("iris")
    second = default_dataset_registry.resolve("iris")
    assert first.identity == "sklearn:iris:v1"
    assert first.task == TaskType.MULTICLASS_CLASSIFICATION
    assert first.features.shape == (150, 4)
    assert first.target.tolist() == [0] * 50 + [1] * 50 + [2] * 50
    assert first.target_labels == ("setosa", "versicolor", "virginica")
    first.features.iloc[0, 0] = -999
    assert second.features.iloc[0, 0] != -999


def test_string_multiclass_encoding_is_deterministic_across_row_order(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": ["zebra", "ant", "moose", "zebra", "moose", "ant"],
        }
    )
    config = _file_config(tmp_path / "unused.csv", TaskType.MULTICLASS_CLASSIFICATION)
    first = dataset_from_frame(frame, config, "first")
    reordered = frame.iloc[[2, 0, 5, 1, 4, 3]].reset_index(drop=True)
    second = dataset_from_frame(reordered, config, "second")
    assert first.target_labels == second.target_labels == ("ant", "moose", "zebra")
    mapping = dict(zip(frame["target"], first.target, strict=True))
    reordered_mapping = dict(zip(reordered["target"], second.target, strict=True))
    assert mapping == reordered_mapping == {"ant": 0, "moose": 1, "zebra": 2}


def test_multiclass_target_rejects_missing_values(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "target": ["ant", "moose", None, "zebra"]}
    )
    with pytest.raises(ValueError, match="contains missing values"):
        dataset_from_frame(
            frame,
            _file_config(tmp_path / "unused.csv", TaskType.MULTICLASS_CLASSIFICATION),
            "identity",
        )


@pytest.mark.parametrize(
    ("task", "target", "message"),
    [
        (TaskType.MULTICLASS_CLASSIFICATION, ["a", "a", "b", "b"], "at least 3"),
        (TaskType.BINARY_CLASSIFICATION, ["a", "b", "c"], "exactly 2"),
    ],
)
def test_classification_target_cardinality_is_explicit(
    task: TaskType, target: list[str], message: str, tmp_path: Path
) -> None:
    frame = pd.DataFrame({"feature": range(len(target)), "target": target})
    with pytest.raises(ValueError, match=message):
        dataset_from_frame(frame, _file_config(tmp_path / "unused.csv", task), "identity")


def test_multiclass_stratification_is_deterministic_and_nested() -> None:
    dataset = default_dataset_registry.resolve("iris")
    split = SplitConfig(strategy="stratified_kfold", n_splits=3)
    first = build_folds(dataset.target, split, 17, dataset.task)
    second = build_folds(dataset.target, split, 17, dataset.task)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.validation_indices, right.validation_indices)
        assert set(dataset.target.iloc[left.validation_indices]) == {0, 1, 2}
        inner = build_inner_folds(dataset.target, left, split, 31, dataset.task)
        outer_train = set(left.train_indices)
        outer_validation = set(left.validation_indices)
        for inner_fold in inner:
            assert set(inner_fold.train_indices) <= outer_train
            assert set(inner_fold.validation_indices) <= outer_train
            assert not (set(inner_fold.train_indices) & outer_validation)
            assert not (set(inner_fold.validation_indices) & outer_validation)


def test_multiclass_stratification_requires_enough_rows_per_class() -> None:
    target = pd.Series([0, 0, 0, 1, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="each target class needs at least 3"):
        build_folds(
            target,
            SplitConfig(strategy="stratified_kfold", n_splits=3),
            4,
            TaskType.MULTICLASS_CLASSIFICATION,
        )


def test_multiclass_metrics_and_compatibility() -> None:
    truth = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    prediction = np.asarray([0, 2, 2, 0, 1, 1], dtype=np.int64)
    values = compute_metrics(
        (MetricName.ACCURACY, MetricName.BALANCED_ACCURACY, MetricName.F1_MACRO),
        truth,
        prediction,
        None,
        TaskType.MULTICLASS_CLASSIFICATION,
    )
    assert values == pytest.approx(
        {"accuracy": 4 / 6, "balanced_accuracy": 2 / 3, "f1_macro": 2 / 3}
    )
    assert metric_spec(MetricName.ACCURACY).supports_task(TaskType.BINARY_CLASSIFICATION)
    assert metric_spec(MetricName.ACCURACY).supports_task(
        TaskType.MULTICLASS_CLASSIFICATION
    )
    assert not metric_spec(MetricName.F1).supports_task(TaskType.MULTICLASS_CLASSIFICATION)
    with pytest.raises(ValueError, match="incompatible"):
        compute_metrics(
            (MetricName.F1,), truth, prediction, None, TaskType.MULTICLASS_CLASSIFICATION
        )


def test_model_capabilities_and_identity_cover_both_classification_tasks() -> None:
    classifiers = {
        "logistic_regression",
        "random_forest_classifier",
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
        "dummy_classifier",
    }
    for name in classifiers:
        assert default_model_registry.supports_task(name, TaskType.BINARY_CLASSIFICATION)
        assert default_model_registry.supports_task(name, TaskType.MULTICLASS_CLASSIFICATION)
        assert not default_model_registry.supports_task(name, TaskType.REGRESSION)
    assert default_model_registry.supported_tasks("ridge_regressor") == frozenset(
        {TaskType.REGRESSION}
    )
    first = default_model_registry.identity_for(classifiers)
    second = default_model_registry.identity_for(set(reversed(sorted(classifiers))))
    assert first == second
    assert first["logistic_regression"]["supported_tasks"] == [
        "binary_classification",
        "multiclass_classification",
    ]
    with pytest.raises(ValueError, match="not configured task"):
        default_model_registry.create(
            ModelConfig(name="ridge_regressor"), 42, TaskType.MULTICLASS_CLASSIFICATION
        )


@pytest.mark.parametrize(
    ("strategy", "metrics"),
    [
        ("kfold", ["accuracy"]),
        ("stratified_kfold", ["f1"]),
        ("stratified_kfold", ["root_mean_squared_error"]),
        ("stratified_kfold", ["roc_auc"]),
    ],
)
def test_invalid_multiclass_run_semantics_fail_validation(
    strategy: str, metrics: list[str]
) -> None:
    data = load_run_config("configs/examples/iris_multiclass_run.yaml").model_dump(mode="json")
    data["split"]["strategy"] = strategy
    data["metrics"] = metrics
    with pytest.raises(ValidationError, match="incompatible"):
        RunConfig.model_validate(data)


def test_binary_f1_and_regression_metrics_are_unchanged() -> None:
    binary = compute_metrics(
        (MetricName.F1,),
        np.asarray([0, 1, 1, 0]),
        np.asarray([0, 1, 0, 0]),
        None,
        TaskType.BINARY_CLASSIFICATION,
    )
    regression = compute_metrics(
        (MetricName.MEAN_ABSOLUTE_ERROR,),
        np.asarray([1.0, 2.0]),
        np.asarray([2.0, 2.0]),
        None,
        TaskType.REGRESSION,
    )
    assert binary["f1"] == pytest.approx(2 / 3)
    assert regression["mean_absolute_error"] == pytest.approx(0.5)
