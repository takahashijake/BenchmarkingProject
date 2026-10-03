import optuna
import pytest

from benchforge.core.config import ModelConfig
from benchforge.models import default_model_registry
from benchforge.search import default_search_space_registry


@pytest.mark.parametrize(
    ("name", "values"),
    [
        (
            "logistic_regression",
            {"C": 1.0, "penalty": "l2", "class_weight": None},
        ),
        (
            "random_forest_classifier",
            {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
            },
        ),
        (
            "extra_trees_classifier",
            {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
            },
        ),
        (
            "hist_gradient_boosting_classifier",
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
def test_default_spaces_propose_estimator_parameters(name: str, values: dict[str, object]) -> None:
    space = default_search_space_registry.resolve(name)
    proposed = space.propose(optuna.trial.FixedTrial(values), {})
    estimator = default_model_registry.create(
        ModelConfig(name=name, parameters={**space.defaults, **proposed}), seed=1
    )
    assert estimator.get_params()


def test_fixed_parameters_are_removed_from_the_search_space() -> None:
    space = default_search_space_registry.resolve("logistic_regression")
    fixed = {"C": 2.5, "penalty": "l2", "class_weight": "balanced"}
    proposed = space.propose(optuna.trial.FixedTrial({}), fixed)
    assert proposed == {}
    assert {**space.defaults, **fixed, **proposed}["C"] == 2.5
