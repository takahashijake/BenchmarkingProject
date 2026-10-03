from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge

from benchforge.core.config import ModelConfig, OutputConfig, load_benchmark_config, load_run_config
from benchforge.data import default_dataset_registry
from benchforge.execution import run_benchmark, run_benchmark_suite
from benchforge.models import default_model_registry
from benchforge.splits import build_folds, build_inner_folds
from benchforge.storage import read_benchmark_artifacts, read_run_artifacts


def test_kfold_is_deterministic_complete_immutable_and_nested() -> None:
    dataset = default_dataset_registry.resolve("diabetes")
    config = load_run_config("configs/examples/diabetes_regression_run.yaml")
    first = build_folds(dataset.target, config.split, 19, config.task)
    second = build_folds(dataset.target, config.split, 19, config.task)
    assert all(
        np.array_equal(left.validation_indices, right.validation_indices)
        for left, right in zip(first, second, strict=True)
    )
    assert sorted(np.concatenate([fold.validation_indices for fold in first])) == list(
        range(dataset.row_count)
    )
    assert all(not fold.train_indices.flags.writeable for fold in first)
    for outer in first:
        inner = build_inner_folds(dataset.target, outer, config.split, 31, config.task)
        outer_train = set(outer.train_indices)
        outer_validation = set(outer.validation_indices)
        for fold in inner:
            assert set(fold.train_indices) < outer_train
            assert set(fold.validation_indices) < outer_train
            assert outer_validation.isdisjoint(fold.train_indices)
            assert outer_validation.isdisjoint(fold.validation_indices)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("linear_regression", LinearRegression),
        ("ridge_regressor", Ridge),
        ("random_forest_regressor", RandomForestRegressor),
        ("extra_trees_regressor", ExtraTreesRegressor),
        ("hist_gradient_boosting_regressor", HistGradientBoostingRegressor),
        ("dummy_regressor", DummyRegressor),
    ],
)
def test_regression_models_resolve_and_run(name: str, expected: type[object]) -> None:
    estimator = default_model_registry.create(
        ModelConfig(name=name),
        seed=7,
        task=load_run_config("configs/examples/diabetes_regression_run.yaml").task,
    )
    assert isinstance(estimator, expected)
    base = load_run_config("configs/examples/diabetes_regression_run.yaml")
    config = base.model_copy(
        update={
            "split": base.split.model_copy(update={"n_splits": 2}),
            "model": ModelConfig(
                name=name,
                parameters={"n_estimators": 20} if "trees" in name or "forest" in name else {},
            ),
        }
    )
    result = run_benchmark(config, persist=False)
    assert len(result.predictions) == 442
    assert all(
        isinstance(item.truth, float) and isinstance(item.prediction, float)
        for item in result.predictions
    )
    assert all(item.score is None for item in result.predictions)


@pytest.mark.parametrize(
    ("config_path", "wrong_model"),
    [
        ("configs/examples/diabetes_regression_run.yaml", "random_forest_classifier"),
        ("configs/examples/breast_cancer_logreg.yaml", "random_forest_regressor"),
    ],
)
def test_wrong_model_family_fails_clearly(config_path: str, wrong_model: str) -> None:
    config = load_run_config(config_path).model_copy(
        update={"model": ModelConfig(name=wrong_model)}
    )
    with pytest.raises(RuntimeError, match="supports task"):
        run_benchmark(config, persist=False)


def test_regression_run_artifacts_round_trip(tmp_path: Path) -> None:
    config = load_run_config("configs/examples/diabetes_regression_run.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path / "runs")}
    )
    result = run_benchmark(config)
    assert result.artifact_directory is not None
    artifacts = read_run_artifacts(result.artifact_directory)
    assert artifacts["metadata"]["dataset_summary"]["task"] == "regression"
    assert artifacts["metadata"]["dataset_summary"]["target_statistics"]["unique_values"] > 2
    assert "." in artifacts["predictions"][0]["prediction"]


@pytest.mark.parametrize(
    ("primary", "reverse"),
    [("root_mean_squared_error", False), ("r2", True)],
)
def test_regression_benchmark_shared_folds_and_direction(
    primary: str, reverse: bool, tmp_path: Path
) -> None:
    config = load_benchmark_config("configs/examples/diabetes_regression_suite.yaml")
    models = (config.models[0], config.models[1], config.models[2])
    config = config.model_copy(
        update={
            "models": models,
            "split": config.split.model_copy(update={"n_splits": 3}),
            "primary_metric": type(config.primary_metric)(primary),
            "output": OutputConfig(directory=tmp_path / primary),
        }
    )
    result = run_benchmark_suite(config)
    fold_maps = [
        {(item.sample_index, item.fold_id) for item in candidate.run_result.predictions}
        for candidate in result.candidates
    ]
    assert all(fold_map == fold_maps[0] for fold_map in fold_maps[1:])
    means = [entry.primary_metric_mean for entry in result.leaderboard]
    assert means == sorted(means, reverse=reverse)
    assert result.artifact_directory is not None
    assert (
        read_benchmark_artifacts(result.artifact_directory)["dataset_summary"]["task"]
        == "regression"
    )
