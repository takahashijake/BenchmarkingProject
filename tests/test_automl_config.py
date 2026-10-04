from pathlib import Path

import pytest
from pydantic import ValidationError

from benchforge.core.config import AutoMLConfig, load_automl_config


@pytest.mark.parametrize(
    "path",
    [
        "configs/examples/breast_cancer_automl.yaml",
        "configs/examples/diabetes_regression_automl.yaml",
    ],
)
def test_valid_automl_examples(path: str) -> None:
    config = load_automl_config(path)
    assert config.budget.total_trials == 10
    assert config.primary_metric in config.metrics


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "metrics": ["root_mean_squared_error"],
                "primary_metric": "root_mean_squared_error",
            },
            "incompatible",
        ),
        ({"outer_split": {"strategy": "kfold", "n_splits": 2}}, "incompatible"),
        ({"budget": {"total_trials": 4, "final_search_trials": 4}}, "less than"),
        ({"budget": {"total_trials": 0}}, "greater than 0"),
        (
            {"models": {"include": ["logistic_regression"], "exclude": ["logistic_regression"]}},
            "overlap",
        ),
        ({"models": {"include": ["logistic_regression", "logistic_regression"]}}, "unique"),
    ],
)
def test_invalid_automl_config_fails(change: dict[str, object], message: str) -> None:
    data = load_automl_config(
        "configs/examples/breast_cancer_automl.yaml"
    ).model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError, match=message):
        AutoMLConfig.model_validate(data)


def test_output_path_is_excluded_from_policy_identity() -> None:
    config = load_automl_config(Path("configs/examples/breast_cancer_automl.yaml"))
    moved = config.model_copy(
        update={"output": config.output.model_copy(update={"directory": Path("elsewhere")})}
    )
    assert moved.fingerprint == config.fingerprint
