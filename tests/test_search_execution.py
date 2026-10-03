from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchforge.core.config import DatasetConfig, OutputConfig, SearchConfig, load_search_config
from benchforge.data.registry import Dataset, DatasetRegistry
from benchforge.search import run_search
from benchforge.search.registry import SearchSpaceRegistry
from benchforge.search.runner import FamilySearchError
from benchforge.search.spaces import SearchSpace
from benchforge.storage import read_search_artifacts


def _fast_config(*, final: bool = False) -> SearchConfig:
    config = load_search_config("configs/examples/breast_cancer_search.yaml")
    models = tuple(model for model in config.models if model.id in {"logistic", "dummy"})
    return config.model_copy(
        update={
            "models": models,
            "final_search": config.final_search.model_copy(update={"enabled": final, "trials": 2}),
        }
    )


def _semantic_result(result: object) -> tuple[object, ...]:
    search = result
    return (
        search.fingerprint,
        [family.aggregate_metrics for family in search.families],
        [family.predictions for family in search.families],
        [
            [
                (
                    fold.selected_parameters,
                    fold.best_inner_score,
                    [
                        (
                            trial.number,
                            trial.state,
                            trial.proposed_parameters,
                            trial.parameters,
                            trial.value,
                            trial.error_type,
                            trial.error_message,
                        )
                        for trial in fold.trials
                    ],
                )
                for fold in family.outer_folds
            ]
            for family in search.families
        ],
        [
            (entry.rank, entry.model_identifier, entry.primary_metric_mean)
            for entry in search.leaderboard
        ],
    )


def test_search_is_reproducible_and_seed_changes_proposals() -> None:
    config = _fast_config()
    first = run_search(config, persist=False)
    second = run_search(config, persist=False)
    assert _semantic_result(first) == _semantic_result(second)
    changed = run_search(config.model_copy(update={"seed": 43}), persist=False)
    first_trials = first.families[0].outer_folds[0].trials
    changed_trials = changed.families[0].outer_folds[0].trials
    assert [trial.proposed_parameters for trial in first_trials] != [
        trial.proposed_parameters for trial in changed_trials
    ]


def test_nested_fold_boundaries_leaderboard_and_compute_counts() -> None:
    config = _fast_config()
    result = run_search(config, persist=False)
    assert result.outer_fold_count == 2
    validation_maps = {
        family.model_identifier: [fold.outer_validation_indices for fold in family.outer_folds]
        for family in result.families
    }
    assert len({str(value) for value in validation_maps.values()}) == 1

    for family in result.families:
        assert family.duration_seconds >= 0
        assert {prediction.sample_index for prediction in family.predictions} == set(range(569))
        assert len(family.predictions) == 569
        for fold in family.outer_folds:
            assert fold.duration_seconds >= 0
            outer_train = set(fold.outer_train_indices)
            outer_validation = set(fold.outer_validation_indices)
            assert outer_train.isdisjoint(outer_validation)
            for inner in fold.inner_folds:
                assert set(inner.train_indices) < outer_train
                assert set(inner.validation_indices) < outer_train
                assert set(inner.train_indices).isdisjoint(outer_validation)
                assert set(inner.validation_indices).isdisjoint(outer_validation)
            assert all(trial.duration_seconds >= 0 for trial in fold.trials)

    logistic = next(family for family in result.families if family.model_identifier == "logistic")
    dummy = next(family for family in result.families if family.model_identifier == "dummy")
    assert logistic.configured_trials == logistic.completed_trials == 4
    assert dummy.configured_trials == dummy.completed_trials == 0
    assert all(not fold.trials and not fold.inner_folds for fold in dummy.outer_folds)
    successful = [entry for entry in result.leaderboard if entry.status == "success"]
    assert [entry.primary_metric_mean for entry in successful] == sorted(
        (entry.primary_metric_mean for entry in successful), reverse=True
    )
    for entry in successful:
        family = next(
            family
            for family in result.families
            if family.model_identifier == entry.model_identifier
        )
        assert entry.primary_metric_mean == family.aggregate_metrics[result.primary_metric].mean
    assert result.total_duration_seconds >= 0


def test_outer_evaluation_occurs_once_per_family_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchforge.search.runner as search_runner

    calls: list[int] = []
    original = search_runner.execute_fold

    def observed(*args: object, **kwargs: object) -> object:
        fold = args[2]
        calls.append(fold.fold_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(search_runner, "execute_fold", observed)
    result = run_search(_fast_config(), persist=False)
    assert calls == [0, 1, 0, 1]
    assert len(result.families) == 2


def test_failed_trial_is_recorded_without_killing_family() -> None:
    def sometimes_invalid(trial: object, fixed: object) -> dict[str, object]:
        del fixed
        return {"C": -1.0 if trial.number == 0 else 1.0}

    space = SearchSpace(
        "logistic_regression",
        99,
        {"test": "one recoverable failure"},
        sometimes_invalid,
        defaults={"solver": "liblinear", "max_iter": 1000},
    )
    result = run_search(
        _fast_config().model_copy(update={"models": (_fast_config().models[0],)}),
        search_space_registry=SearchSpaceRegistry({"logistic_regression": space}),
        persist=False,
    )
    assert not result.failures
    assert result.families[0].failed_trials == 2
    assert result.families[0].completed_trials == 2
    failed = [
        trial
        for fold in result.families[0].outer_folds
        for trial in fold.trials
        if trial.state == "FAIL"
    ]
    assert all(trial.error_type and trial.error_message for trial in failed)


def test_systemic_family_failure_is_visible_and_persisted(tmp_path: Path) -> None:
    def always_invalid(trial: object, fixed: object) -> dict[str, object]:
        del trial, fixed
        return {"C": -1.0}

    config = _fast_config().model_copy(
        update={"output": OutputConfig(directory=tmp_path / "artifacts")}
    )
    space = SearchSpace(
        "logistic_regression",
        100,
        {"test": "all fail"},
        always_invalid,
        defaults={"solver": "liblinear", "max_iter": 1000},
    )
    result = run_search(
        config,
        search_space_registry=SearchSpaceRegistry({"logistic_regression": space}),
    )
    assert len(result.failures) == 1
    assert result.failures[0].model_identifier == "logistic"
    assert "all 2 trials failed" in result.failures[0].message
    assert result.failures[0].configured_trials == 2
    assert result.failures[0].failed_trials == 2
    assert result.leaderboard[0].model_identifier == "dummy"
    assert result.artifact_directory is not None
    failed_directory = result.artifact_directory / "families" / "logistic"
    assert {path.name for path in failed_directory.iterdir()} == {
        "failure.json",
        "trials.json",
        "trials.csv",
    }
    only_search = config.model_copy(update={"models": (config.models[0],)})
    with pytest.raises(FamilySearchError, match="every search family failed"):
        run_search(
            only_search,
            search_space_registry=SearchSpaceRegistry({"logistic_regression": space}),
            persist=False,
        )


def test_final_search_follows_and_does_not_change_outer_ranking() -> None:
    without_final = run_search(_fast_config(final=False), persist=False)
    with_final = run_search(_fast_config(final=True), persist=False)
    assert [entry.model_identifier for entry in with_final.leaderboard] == [
        entry.model_identifier for entry in without_final.leaderboard
    ]
    assert [entry.primary_metric_mean for entry in with_final.leaderboard] == [
        entry.primary_metric_mean for entry in without_final.leaderboard
    ]
    assert with_final.final_candidate is not None
    assert with_final.final_candidate.model_identifier == with_final.leaderboard[0].model_identifier
    assert with_final.final_candidate.configured_trials == 2


def test_search_artifacts_are_auditable_and_readable(tmp_path: Path) -> None:
    config = _fast_config(final=True).model_copy(
        update={"output": OutputConfig(directory=tmp_path / "artifacts")}
    )
    result = run_search(config)
    assert result.artifact_directory is not None
    expected = {
        "search_config.json",
        "metadata.json",
        "dataset_summary.json",
        "tuned_leaderboard.json",
        "tuned_leaderboard.csv",
        "failures.json",
        "final_candidate.json",
        "outer_folds.json",
        "families",
    }
    assert {path.name for path in result.artifact_directory.iterdir()} == expected
    artifacts = read_search_artifacts(result.artifact_directory)
    assert artifacts["metadata"]["search_fingerprint"] == result.fingerprint
    assert (
        artifacts["final_candidate"]["model_identifier"] == result.leaderboard[0].model_identifier
    )
    logistic_fold = result.artifact_directory / "families" / "logistic" / "outer_fold_0"
    assert {path.name for path in logistic_fold.iterdir()} == {
        "split_plan.json",
        "trials.json",
        "trials.csv",
        "best_params.json",
        "outer_result.json",
    }


def test_outer_predictions_are_consistent_with_fold_plan() -> None:
    result = run_search(_fast_config(), persist=False)
    for family in result.families:
        expected = np.concatenate([fold.validation_indices for fold in result.outer_folds]).tolist()
        assert sorted(prediction.sample_index for prediction in family.predictions) == sorted(
            expected
        )


def test_sentinel_outer_validation_rows_never_enter_inner_search() -> None:
    rows = 24
    features = pd.DataFrame(
        {
            "sentinel_sample_id": np.arange(rows, dtype=float),
            "signal": [float(index % 5) for index in range(rows)],
        }
    )
    dataset = Dataset(
        identity="test:sentinel:v1",
        task=_fast_config().task,
        features=features,
        target=pd.Series([index % 2 for index in range(rows)]),
        feature_names=("sentinel_sample_id", "signal"),
        numeric_feature_names=("sentinel_sample_id", "signal"),
        categorical_feature_names=(),
        target_name="target",
        missing_values={"sentinel_sample_id": 0, "signal": 0},
        target_labels=("0", "1"),
    )
    registry = DatasetRegistry({"sentinel": lambda: dataset})
    base = _fast_config()
    config = base.model_copy(
        update={
            "dataset": DatasetConfig(name="sentinel"),
            "models": (base.models[0],),
        }
    )
    result = run_search(config, dataset_registry=registry, persist=False)
    family = result.families[0]
    for fold in family.outer_folds:
        sentinel_ids = set(features.iloc[list(fold.outer_validation_indices)]["sentinel_sample_id"])
        objective_ids = {
            float(features.iloc[index]["sentinel_sample_id"])
            for inner in fold.inner_folds
            for index in np.concatenate([inner.train_indices, inner.validation_indices])
        }
        assert sentinel_ids.isdisjoint(objective_ids)


def test_mixed_csv_dataset_runs_through_search_path() -> None:
    config = load_search_config("configs/examples/mixed_tabular_search.yaml")
    result = run_search(config, persist=False)
    assert result.dataset.row_count == 30
    assert result.dataset.categorical_features == ("segment", "uses_mobile")
    assert all(len(family.predictions) == 30 for family in result.families)
