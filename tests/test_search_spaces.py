import optuna
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler

import benchforge.models.registry as model_registry_module
from benchforge.core.config import ModelConfig, TaskType
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


def test_fixed_l1_penalty_replaces_only_the_unfixed_default_solver() -> None:
    space = default_search_space_registry.resolve("logistic_regression")
    fixed = {"C": 2.5, "penalty": "l1", "class_weight": "balanced"}
    proposed = space.propose(optuna.trial.FixedTrial({}), fixed)
    parameters = {**space.defaults, **fixed, **proposed}
    assert proposed == {"solver": "saga"}
    assert parameters["penalty"] == "l1"
    assert parameters["solver"] == "saga"


def test_explicit_solver_still_wins_when_penalty_is_fixed() -> None:
    space = default_search_space_registry.resolve("logistic_regression")
    fixed = {
        "solver": "liblinear",
        "C": 2.5,
        "penalty": "l1",
        "class_weight": "balanced",
    }
    assert space.propose(optuna.trial.FixedTrial({}), fixed) == {}


@pytest.mark.parametrize(
    ("solver", "trial_values", "expected"),
    [
        ("lbfgs", {}, {}),
        ("newton-cg", {}, {}),
        ("newton-cholesky", {}, {}),
        ("sag", {}, {}),
        ("liblinear", {"penalty": "l1"}, {"penalty": "l1"}),
        ("saga", {"penalty": "l1"}, {"penalty": "l1"}),
    ],
)
def test_logistic_penalty_proposals_respect_fixed_solver(
    solver: str,
    trial_values: dict[str, object],
    expected: dict[str, object],
) -> None:
    space = default_search_space_registry.resolve("logistic_regression")
    fixed = {"solver": solver, "C": 1.0, "class_weight": None}
    assert space.propose(optuna.trial.FixedTrial(trial_values), fixed) == expected


@pytest.mark.parametrize(
    ("task", "loader"),
    [
        (TaskType.BINARY_CLASSIFICATION, load_breast_cancer),
        (TaskType.MULTICLASS_CLASSIFICATION, load_iris),
    ],
)
def test_default_logistic_search_configuration_fits_supported_classification_tasks(
    task: TaskType, loader: object
) -> None:
    space = default_search_space_registry.resolve("logistic_regression")
    proposed = space.propose(
        optuna.trial.FixedTrial({"C": 1.0, "class_weight": None}), {}
    )
    estimator = default_model_registry.create(
        ModelConfig(name="logistic_regression", parameters={**space.defaults, **proposed}),
        seed=1,
        task=task,
    )
    features, target = loader(return_X_y=True)  # type: ignore[operator]
    features = StandardScaler().fit_transform(features)
    estimator.fit(features, target)
    assert len(estimator.predict(features[:3])) == 3


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.7.2", {"penalty": "l1"}),
        ("1.8.0", {"l1_ratio": 1.0}),
    ],
)
def test_logistic_penalty_translation_tracks_sklearn_api(
    version: str, expected: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_registry_module, "sklearn_version", version)
    estimator = model_registry_module._logistic_regression(
        {"solver": "saga", "penalty": "l1"}, seed=1
    )
    parameters = estimator.get_params()
    assert all(parameters[name] == value for name, value in expected.items())
    if version >= "1.8":
        assert parameters["penalty"] == "deprecated"


def test_logistic_search_identity_records_corrected_solver_semantics() -> None:
    identity = default_search_space_registry.resolve("logistic_regression").identity
    assert identity["version"] == 3
    assert identity["definition"]["penalty"]["l1_capable_solvers"] == [
        "liblinear",
        "saga",
    ]
    assert identity["definition"]["penalty"]["solver_for_fixed_l1_or_elasticnet"] == "saga"
