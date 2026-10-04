from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchforge.automl import build_automl_plan, run_automl
from benchforge.cli import main
from benchforge.core.config import (
    AutoMLModelPolicy,
    ModelConfig,
    OutputConfig,
    SearchModelConfig,
    load_automl_config,
    load_benchmark_config,
    load_run_config,
    load_search_config,
)
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.runner import run_benchmark
from benchforge.search.runner import run_search
from benchforge.storage import (
    read_automl_artifacts,
    read_benchmark_artifacts,
    read_run_artifacts,
    read_search_artifacts,
)


def _without_runtime(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _without_runtime(asdict(value))
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in value.items()
            if "duration" not in key and key != "artifact_directory"
        }
    if isinstance(value, tuple):
        return tuple(_without_runtime(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def test_multiclass_run_is_deterministic_and_persists_integer_predictions(
    tmp_path: Path,
) -> None:
    config = load_run_config("configs/examples/iris_multiclass_run.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path)}
    )
    first = run_benchmark(config, persist=False)
    second = run_benchmark(config)
    assert first.fingerprint == second.fingerprint
    assert first.aggregate_metrics == second.aggregate_metrics
    assert first.predictions == second.predictions
    assert all(record.score is None for record in first.predictions)
    assert all(isinstance(record.truth, int) for record in first.predictions)
    assert all(isinstance(record.prediction, int) for record in first.predictions)
    assert second.artifact_directory is not None
    artifacts = read_run_artifacts(second.artifact_directory)
    assert artifacts["metadata"]["dataset_summary"]["target_labels"] == [
        "setosa",
        "versicolor",
        "virginica",
    ]


def test_multiclass_benchmark_uses_shared_folds_and_persists(tmp_path: Path) -> None:
    config = load_benchmark_config("configs/examples/iris_multiclass_suite.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path)}
    )
    result = run_benchmark_suite(config)
    validation_assignments = [
        tuple(record.fold_id for record in candidate.run_result.predictions)
        for candidate in result.candidates
    ]
    assert all(item == validation_assignments[0] for item in validation_assignments)
    successful = [entry for entry in result.leaderboard if entry.status == "success"]
    assert [entry.rank for entry in successful] == list(range(1, len(successful) + 1))
    assert successful[0].primary_metric_mean >= successful[-1].primary_metric_mean
    assert result.artifact_directory is not None
    artifacts = read_benchmark_artifacts(result.artifact_directory)
    assert artifacts["benchmark_config"]["task"] == "multiclass_classification"


def test_multiclass_benchmark_isolates_an_incompatible_regressor() -> None:
    base = load_benchmark_config("configs/examples/iris_multiclass_suite.yaml")
    config = base.model_copy(
        update={
            "models": (
                ModelConfig(name="ridge_regressor", id="invalid_regressor"),
                ModelConfig(name="dummy_classifier", id="valid_dummy"),
            )
        }
    )
    result = run_benchmark_suite(config, persist=False)
    assert [candidate.model_identifier for candidate in result.candidates] == ["valid_dummy"]
    assert [failure.model_identifier for failure in result.failures] == ["invalid_regressor"]
    assert "not configured task" in result.failures[0].message


def test_multiclass_nested_search_is_deterministic_leakage_safe_and_persists(
    tmp_path: Path,
) -> None:
    config = load_search_config("configs/examples/iris_multiclass_search.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path)}
    )
    first = run_search(config, persist=False)
    second = run_search(config)
    assert first.fingerprint == second.fingerprint
    assert _without_runtime(first.families) == _without_runtime(second.families)
    assert _without_runtime(first.leaderboard) == _without_runtime(second.leaderboard)
    assert _without_runtime(first.final_candidate) == _without_runtime(second.final_candidate)
    for family in first.families:
        assert tuple(fold.outer_validation_indices for fold in family.outer_folds) == tuple(
            tuple(int(index) for index in fold.validation_indices) for fold in first.outer_folds
        )
        for fold in family.outer_folds:
            outer_train = set(fold.outer_train_indices)
            outer_validation = set(fold.outer_validation_indices)
            for inner in fold.inner_folds:
                assert set(inner.train_indices) <= outer_train
                assert set(inner.validation_indices) <= outer_train
                assert not (set(inner.train_indices) & outer_validation)
                assert not (set(inner.validation_indices) & outer_validation)
    assert second.final_candidate is not None
    assert second.artifact_directory is not None
    artifacts = read_search_artifacts(second.artifact_directory)
    assert artifacts["dataset_summary"]["task"] == "multiclass_classification"
    assert artifacts["final_candidate"] is not None


def test_multiclass_search_isolates_a_failing_family() -> None:
    base = load_search_config("configs/examples/iris_multiclass_search.yaml")
    config = base.model_copy(
        update={
            "models": (
                SearchModelConfig(
                    name="logistic_regression",
                    id="broken_logistic",
                    mode="search",
                    parameters={"solver": "not-a-solver"},
                ),
                SearchModelConfig(
                    name="dummy_classifier",
                    id="dummy",
                    mode="fixed",
                    parameters={"strategy": "prior"},
                ),
            ),
            "final_search": base.final_search.model_copy(update={"enabled": False}),
        }
    )
    result = run_search(config, persist=False)
    assert [failure.model_identifier for failure in result.failures] == ["broken_logistic"]
    assert [family.model_identifier for family in result.families] == ["dummy"]


def test_multiclass_automl_discovery_plan_execution_and_artifact_composition(
    tmp_path: Path,
) -> None:
    base = load_automl_config("configs/examples/iris_multiclass_automl.yaml")
    first_plan = build_automl_plan(base)
    second_plan = build_automl_plan(base)
    assert first_plan == second_plan
    assert first_plan.searchable_families == (
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
        "logistic_regression",
        "random_forest_classifier",
    )
    assert first_plan.fixed_families == ("dummy_classifier",)
    assert all("regressor" not in name for name in first_plan.searchable_families)
    assert first_plan.search_config.task.value == "multiclass_classification"

    config = base.model_copy(
        update={
            "models": AutoMLModelPolicy(
                include=("logistic_regression", "dummy_classifier"),
                include_fixed_baselines=True,
            ),
            "budget": base.budget.model_copy(
                update={"total_trials": 2, "final_search_trials": 0}
            ),
            "output": OutputConfig(directory=tmp_path),
        }
    )
    result = run_automl(config)
    assert result.plan.searchable_families == ("logistic_regression",)
    assert result.plan.fixed_families == ("dummy_classifier",)
    assert result.artifact_directory is not None
    assert result.search_result.artifact_directory == result.artifact_directory / "search"
    artifacts = read_automl_artifacts(result.artifact_directory)
    assert artifacts["generated_search_config"]["task"] == "multiclass_classification"
    assert artifacts["result"]["search_artifacts"] == "search"


def test_multiclass_cli_smoke_for_all_workflow_levels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = [
        ("run", "configs/examples/iris_multiclass_run.yaml"),
        ("benchmark", "configs/examples/iris_multiclass_suite.yaml"),
        ("search", "configs/examples/iris_multiclass_search.yaml"),
    ]
    for command, source in cases:
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        data["output"]["directory"] = str(tmp_path / command)
        if command == "benchmark":
            data["models"] = data["models"][:2]
        if command == "search":
            data["models"] = [data["models"][0], data["models"][-1]]
            data["final_search"]["enabled"] = False
        path = tmp_path / f"{command}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert main([command, str(path)]) == 0

    assert main(
        ["automl", "configs/examples/iris_multiclass_automl.yaml", "--plan-only"]
    ) == 0
    output = capsys.readouterr().out
    assert "BenchForge run complete" in output
    assert "BenchForge benchmark complete" in output
    assert "BenchForge nested search complete" in output
    assert "BenchForge AutoML plan" in output
