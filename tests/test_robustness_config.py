from pathlib import Path

import pytest
from pydantic import ValidationError

from benchforge import RobustnessConfig, load_robustness_config


@pytest.mark.parametrize("name", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_examples_parse_and_round_trip(name: str) -> None:
    config = load_robustness_config(f"configs/examples/{name}_robustness.yaml")
    assert config.robustness.repetitions == 5
    restored = RobustnessConfig.model_validate_json(config.canonical_json())
    assert restored.canonical_json() == config.canonical_json()


@pytest.mark.parametrize(
    "change",
    [
        {"models": []},
        {"models": [{"name": "dummy_classifier"}]},
        {
            "models": [
                {"name": "dummy_classifier", "id": "same"},
                {"name": "logistic_regression", "id": "same"},
            ]
        },
        {"robustness": {"repetitions": 1}},
        {"robustness": {"repetitions": True}},
        {"robustness": {"repetitions": 2.5}},
        {"robustness": {"confidence_level": 0}},
        {"robustness": {"confidence_level": 1}},
        {"robustness": {"confidence_level": float("nan")}},
        {"robustness": {"confidence_level": float("inf")}},
        {"robustness": {"bootstrap_samples": 99}},
        {"robustness": {"bootstrap_samples": 100001}},
        {"robustness": {"unexpected": True}},
        {"metrics": ["root_mean_squared_error"], "primary_metric": "root_mean_squared_error"},
        {"metrics": ["f1_macro"], "primary_metric": "f1_macro"},
        {"metrics": []},
        {"metrics": ["accuracy", "accuracy"], "primary_metric": "accuracy"},
        {"primary_metric": "accuracy"},
        {"models": [{"name": "ridge_regressor"}, {"name": "dummy_classifier"}]},
        {"models": [{"name": "unknown"}, {"name": "dummy_classifier"}]},
        {
            "models": [
                {"name": "dummy_classifier", "parameters": {"random_state": 1}},
                {"name": "logistic_regression"},
            ]
        },
        {"models": [{"name": "dummy_classifier", "id": ".."}, {"name": "logistic_regression"}]},
        {
            "models": [
                {"name": "logistic_regression", "parameters": {"C": float("nan")}},
                {"name": "dummy_classifier"},
            ]
        },
        {"split": {"strategy": "kfold", "n_splits": 3}},
        {"split": {"strategy": "holdout", "n_splits": 3}},
        {"split": {"strategy": "stratified_kfold", "n_splits": 1}},
        {"split": {"strategy": "stratified_kfold", "n_splits": 3, "shuffle": False}},
        {"dataset": {"name": "iris"}},
        {"dataset": {"name": "unknown"}},
        {"dataset": {"source": "csv", "path": "x.csv", "target_column": "y", "task": "regression"}},
        {"unexpected": 1},
    ],
)
def test_invalid_configuration(change: dict[str, object]) -> None:
    data = load_robustness_config("configs/examples/breast_cancer_robustness.yaml").model_dump(
        mode="json"
    )
    data.update(change)
    with pytest.raises(ValidationError):
        RobustnessConfig.model_validate(data)


@pytest.mark.parametrize("contents", ["[1, 2]", "invalid: [", ""])
def test_invalid_yaml(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_robustness_config(path)


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read configuration"):
        load_robustness_config(tmp_path / "missing.yaml")
