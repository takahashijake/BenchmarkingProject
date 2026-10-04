from pathlib import Path

import pytest

from benchforge.automl import build_automl_plan
from benchforge.core.config import AutoMLModelPolicy, ModelConfig, OutputConfig, load_automl_config
from benchforge.models import ModelRegistry, default_model_registry


def test_classification_discovery_budget_and_fit_estimate_are_deterministic() -> None:
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    first = build_automl_plan(config)
    second = build_automl_plan(config)
    assert first == second
    assert first.searchable_families == (
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
        "logistic_regression",
        "random_forest_classifier",
    )
    assert first.fixed_families == ("dummy_classifier",)
    assert first.study_count == 8
    assert first.outer_search_budget == 8
    assert first.trials_per_outer_fold == 1
    assert first.allocated_outer_trials == 8
    assert first.allocated_trials == 10
    assert first.unallocated_trials == 0
    assert first.approximate_maximum_fits == 30
    assert first.search_config.budget.trials_per_outer_fold == 1
    assert first.search_config.final_search.enabled
    assert first.search_config.final_search.trials == 2


def test_regression_discovery_includes_registry_fixed_models() -> None:
    plan = build_automl_plan(
        load_automl_config("configs/examples/diabetes_regression_automl.yaml")
    )
    assert plan.searchable_families == (
        "extra_trees_regressor",
        "hist_gradient_boosting_regressor",
        "random_forest_regressor",
        "ridge_regressor",
    )
    assert plan.fixed_families == ("dummy_regressor", "linear_regression")
    assert all(
        candidate.mode == "fixed"
        for candidate in plan.candidates
        if candidate.name in plan.fixed_families
    )
    assert plan.approximate_maximum_fits == 32


def test_include_exclude_and_remainder_policy() -> None:
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    config = config.model_copy(
        update={
            "models": AutoMLModelPolicy(
                include=("logistic_regression", "dummy_classifier"),
                exclude=("random_forest_classifier",),
                include_fixed_baselines=True,
            ),
            "budget": config.budget.model_copy(
                update={"total_trials": 5, "final_search_trials": 2}
            ),
        }
    )
    plan = build_automl_plan(config)
    assert plan.searchable_families == ("logistic_regression",)
    assert plan.fixed_families == ("dummy_classifier",)
    assert plan.trials_per_outer_fold == 1
    assert plan.unallocated_trials == 1


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (AutoMLModelPolicy(include=("not_a_model",)), "unknown AutoML model"),
        (AutoMLModelPolicy(include=("ridge_regressor",)), "incompatible"),
        (
            AutoMLModelPolicy(
                include=("dummy_classifier",), include_fixed_baselines=True
            ),
            "no searchable",
        ),
        (
            AutoMLModelPolicy(
                include=("dummy_classifier",), include_fixed_baselines=False
            ),
            "no searchable",
        ),
    ],
)
def test_invalid_model_policies_fail_before_execution(
    policy: AutoMLModelPolicy, message: str
) -> None:
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml").model_copy(
        update={"models": policy}
    )
    with pytest.raises(ValueError, match=message):
        build_automl_plan(config)


def test_underfunded_budget_fails_and_output_does_not_change_plan_identity() -> None:
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    underfunded = config.model_copy(
        update={"budget": config.budget.model_copy(update={"total_trials": 7})}
    )
    with pytest.raises(ValueError, match="too small"):
        build_automl_plan(underfunded)
    first = build_automl_plan(config)
    moved = build_automl_plan(
        config.model_copy(update={"output": OutputConfig(directory=Path("elsewhere"))})
    )
    assert moved.fingerprint == first.fingerprint
    assert moved.search_fingerprint == first.search_fingerprint


def test_registry_insertion_order_cannot_change_plan_ordering() -> None:
    names = [
        name
        for name in default_model_registry.names
        if default_model_registry.task(name).value == "binary_classification"
    ]

    def factory(model_name: str):
        def create(parameters: dict[str, object], seed: int):
            return default_model_registry.create(
                ModelConfig.model_validate({"name": model_name, "parameters": parameters}),
                seed,
            )

        return create

    forward = ModelRegistry({name: factory(name) for name in names})
    reverse = ModelRegistry({name: factory(name) for name in reversed(names)})
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    assert build_automl_plan(config, model_registry=forward).candidates == build_automl_plan(
        config, model_registry=reverse
    ).candidates
