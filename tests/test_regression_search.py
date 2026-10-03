import csv
from pathlib import Path

import optuna
import pytest

import benchforge.search.runner as search_runner
from benchforge.core.config import MetricName, OutputConfig, load_search_config
from benchforge.models import default_model_registry
from benchforge.search import default_search_space_registry, run_search
from benchforge.storage import read_search_artifacts


def _fast_config(*, metric: MetricName = MetricName.ROOT_MEAN_SQUARED_ERROR, final: bool = False):
    config = load_search_config("configs/examples/diabetes_regression_search.yaml")
    models = tuple(model for model in config.models if model.id in {"ridge", "dummy"})
    return config.model_copy(
        update={
            "models": models,
            "primary_metric": metric,
            "final_search": config.final_search.model_copy(update={"enabled": final, "trials": 2}),
        }
    )


def _semantics(result: object) -> object:
    search = result
    return (
        search.fingerprint,
        [(family.aggregate_metrics, family.predictions) for family in search.families],
        [
            [
                (
                    fold.selected_parameters,
                    fold.best_inner_score,
                    [
                        (trial.proposed_parameters, trial.parameters, trial.value, trial.state)
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


@pytest.mark.parametrize(
    ("metric", "direction", "reverse"),
    [
        (MetricName.ROOT_MEAN_SQUARED_ERROR, "minimize", False),
        (MetricName.R2, "maximize", True),
    ],
)
def test_regression_nested_search_uses_metric_direction(
    metric: MetricName,
    direction: str,
    reverse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions: list[str] = []
    original_create_study = optuna.create_study

    def observed_create_study(*args: object, **kwargs: object):
        directions.append(str(kwargs["direction"]))
        return original_create_study(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", observed_create_study)
    result = run_search(_fast_config(metric=metric), persist=False)
    assert result.optimization_direction == direction
    assert directions and set(directions) == {direction}
    means = [entry.primary_metric_mean for entry in result.leaderboard]
    assert means == sorted(means, reverse=reverse)
    ridge = next(family for family in result.families if family.model_identifier == "ridge")
    dummy = next(family for family in result.families if family.model_identifier == "dummy")
    assert ridge.configured_trials == ridge.completed_trials == 4
    assert dummy.configured_trials == dummy.completed_trials == 0
    assert all(not fold.trials for fold in dummy.outer_folds)
    assert all(len(family.predictions) == 442 for family in result.families)


def test_regression_search_is_reproducible() -> None:
    first = run_search(_fast_config(), persist=False)
    second = run_search(_fast_config(), persist=False)
    assert _semantics(first) == _semantics(second)


def test_final_search_does_not_change_nested_ranking() -> None:
    without = run_search(_fast_config(final=False), persist=False)
    with_final = run_search(_fast_config(final=True), persist=False)
    assert [entry.model_identifier for entry in with_final.leaderboard] == [
        entry.model_identifier for entry in without.leaderboard
    ]
    assert [entry.primary_metric_mean for entry in with_final.leaderboard] == [
        entry.primary_metric_mean for entry in without.leaderboard
    ]
    assert with_final.final_candidate is not None
    assert with_final.final_candidate.model_identifier == with_final.leaderboard[0].model_identifier
    assert with_final.final_candidate.cv_selection_score > 0


def test_regression_family_ranking_uses_outer_metrics_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = search_runner._run_study

    def deliberately_bad_inner_score(*args: object, **kwargs: object):
        selected, _score, trials = original(*args, **kwargs)
        return selected, 1_000_000.0, trials

    monkeypatch.setattr(search_runner, "_run_study", deliberately_bad_inner_score)
    result = run_search(_fast_config(), persist=False)
    ridge = next(family for family in result.families if family.model_identifier == "ridge")
    assert all(fold.best_inner_score == 1_000_000.0 for fold in ridge.outer_folds)
    assert result.leaderboard[0].model_identifier == "ridge"
    assert (
        result.leaderboard[0].primary_metric_mean
        == ridge.aggregate_metrics["root_mean_squared_error"].mean
    )


def test_regression_search_artifacts_round_trip(tmp_path: Path) -> None:
    config = _fast_config(final=True).model_copy(
        update={"output": OutputConfig(directory=tmp_path / "artifacts")}
    )
    result = run_search(config)
    assert result.artifact_directory is not None
    artifacts = read_search_artifacts(result.artifact_directory)
    assert artifacts["metadata"]["optimization_direction"] == "minimize"
    assert artifacts["dataset_summary"]["target_statistics"]["unique_values"] > 2
    predictions = result.artifact_directory / "families" / "ridge" / "outer_predictions.csv"
    with predictions.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["score"] == "" for row in rows)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("ridge_regressor", {"alpha": 1.0}),
        (
            "random_forest_regressor",
            {
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        ),
        (
            "extra_trees_regressor",
            {
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        ),
        (
            "hist_gradient_boosting_regressor",
            {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "max_depth": None,
            },
        ),
    ],
)
def test_regression_search_spaces_build_valid_estimators(
    name: str, values: dict[str, object]
) -> None:
    space = default_search_space_registry.resolve(name)
    proposed = space.propose(optuna.trial.FixedTrial(values), {})
    estimator = default_model_registry.create(
        search_runner.ModelConfig(name=name, parameters={**space.defaults, **proposed}),
        seed=2,
        task=_fast_config().task,
    )
    assert estimator.get_params()


def test_fixed_parameter_collision_policy_for_regression() -> None:
    space = default_search_space_registry.resolve("ridge_regressor")
    assert space.propose(optuna.trial.FixedTrial({}), {"alpha": 2.5}) == {}


def test_regression_outer_validation_never_reaches_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[set[int], ...]] = []
    original = search_runner._run_study

    def inspect_study(*args: object, **kwargs: object):
        folds = args[3]
        observed.append(
            tuple(set(fold.train_indices) | set(fold.validation_indices) for fold in folds)
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(search_runner, "_run_study", inspect_study)
    result = run_search(_fast_config(), persist=False)
    ridge = next(family for family in result.families if family.model_identifier == "ridge")
    assert len(observed) == len(ridge.outer_folds)
    for seen, fold in zip(observed, ridge.outer_folds, strict=True):
        outer_train = set(fold.outer_train_indices)
        outer_validation = set(fold.outer_validation_indices)
        assert all(indices == outer_train for indices in seen)
        assert all(indices.isdisjoint(outer_validation) for indices in seen)
