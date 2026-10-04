from pathlib import Path

import pytest

from benchforge.automl import run_automl
from benchforge.core.config import AutoMLModelPolicy, OutputConfig, load_automl_config
from benchforge.search import default_search_space_registry
from benchforge.search.registry import SearchSpaceRegistry
from benchforge.search.spaces import SearchSpace
from benchforge.storage import read_automl_artifacts, read_search_artifacts


def _fast_config(path: str, *, output: Path | None = None):
    config = load_automl_config(path)
    searchable = (
        "logistic_regression"
        if config.task.value == "binary_classification"
        else "ridge_regressor"
    )
    fixed = (
        "dummy_classifier"
        if config.task.value == "binary_classification"
        else "dummy_regressor"
    )
    updates: dict[str, object] = {
        "models": AutoMLModelPolicy(include=(searchable, fixed)),
        "budget": config.budget.model_copy(
            update={"total_trials": 2, "final_search_trials": 0}
        ),
    }
    if output is not None:
        updates["output"] = OutputConfig(directory=output)
    return config.model_copy(update=updates)


@pytest.mark.parametrize(
    "path",
    [
        "configs/examples/breast_cancer_automl.yaml",
        "configs/examples/diabetes_regression_automl.yaml",
    ],
)
def test_automl_completes_through_existing_nested_search(path: str) -> None:
    result = run_automl(_fast_config(path), persist=False)
    search = result.search_result
    assert search.fingerprint == result.plan.search_fingerprint
    assert search.outer_fold_count == 2
    assert {family.candidate.mode for family in search.families} == {"search", "fixed"}
    assert search.leaderboard == tuple(search.leaderboard)
    searchable = next(family for family in search.families if family.candidate.mode == "search")
    fixed = next(family for family in search.families if family.candidate.mode == "fixed")
    assert searchable.configured_trials == 2
    assert fixed.configured_trials == 0
    assert [fold.validation_indices.tolist() for fold in search.outer_folds] == [
        list(fold.outer_validation_indices) for fold in searchable.outer_folds
    ]
    assert [fold.validation_indices.tolist() for fold in search.outer_folds] == [
        list(fold.outer_validation_indices) for fold in fixed.outer_folds
    ]
    for fold in searchable.outer_folds:
        outer_validation = set(fold.outer_validation_indices)
        assert all(
            set(inner.train_indices).isdisjoint(outer_validation)
            and set(inner.validation_indices).isdisjoint(outer_validation)
            for inner in fold.inner_folds
        )


def test_automl_is_materially_reproducible() -> None:
    config = _fast_config("configs/examples/breast_cancer_automl.yaml")
    first = run_automl(config, persist=False)
    second = run_automl(config, persist=False)
    assert first.fingerprint == second.fingerprint
    assert first.plan == second.plan
    assert [entry.model_identifier for entry in first.search_result.leaderboard] == [
        entry.model_identifier for entry in second.search_result.leaderboard
    ]
    assert [family.aggregate_metrics for family in first.search_result.families] == [
        family.aggregate_metrics for family in second.search_result.families
    ]


def test_automl_artifacts_compose_search_artifacts(tmp_path: Path) -> None:
    config = _fast_config(
        "configs/examples/breast_cancer_automl.yaml", output=tmp_path / "artifacts"
    )
    result = run_automl(config)
    assert result.artifact_directory is not None
    assert result.search_result.artifact_directory == result.artifact_directory / "search"
    assert {path.name for path in result.artifact_directory.iterdir()} == {
        "automl_config.json",
        "plan.json",
        "generated_search_config.json",
        "metadata.json",
        "manifest.json",
        "result.json",
        "search",
    }
    artifacts = read_automl_artifacts(result.artifact_directory)
    nested = read_search_artifacts(result.artifact_directory / "search")
    assert artifacts["metadata"]["automl_fingerprint"] == result.fingerprint
    assert artifacts["plan"]["search_fingerprint"] == result.search_result.fingerprint
    assert nested["metadata"]["search_fingerprint"] == result.search_result.fingerprint


def test_family_failure_is_isolated_by_delegated_search() -> None:
    def always_invalid(trial: object, fixed: object) -> dict[str, object]:
        del trial, fixed
        return {"C": -1.0}

    failing = SearchSpace(
        "logistic_regression",
        99,
        {"test": "AutoML failure isolation"},
        always_invalid,
        defaults={"solver": "liblinear", "max_iter": 1000},
    )
    spaces = SearchSpaceRegistry(
        {
            "logistic_regression": failing,
            "random_forest_classifier": default_search_space_registry.resolve(
                "random_forest_classifier"
            ),
        }
    )
    base = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    config = base.model_copy(
        update={
            "models": AutoMLModelPolicy(
                include=(
                    "logistic_regression",
                    "random_forest_classifier",
                    "dummy_classifier",
                )
            ),
            "budget": base.budget.model_copy(
                update={"total_trials": 4, "final_search_trials": 0}
            ),
        }
    )
    result = run_automl(config, search_space_registry=spaces, persist=False)
    assert [failure.model_identifier for failure in result.search_result.failures] == [
        "logistic_regression"
    ]
    assert {family.model_identifier for family in result.search_result.families} == {
        "random_forest_classifier",
        "dummy_classifier",
    }


def test_final_search_reserve_cannot_change_nested_leaderboard() -> None:
    without = _fast_config("configs/examples/breast_cancer_automl.yaml")
    with_final = without.model_copy(
        update={
            "budget": without.budget.model_copy(
                update={"total_trials": 4, "final_search_trials": 2}
            )
        }
    )
    first = run_automl(without, persist=False).search_result
    second = run_automl(with_final, persist=False).search_result
    assert [(row.model_identifier, row.primary_metric_mean) for row in first.leaderboard] == [
        (row.model_identifier, row.primary_metric_mean) for row in second.leaderboard
    ]
    assert second.final_candidate is not None
