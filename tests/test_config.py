from pathlib import Path

import pytest
from pydantic import ValidationError

from benchforge.core.config import RunConfig, load_run_config


def test_example_configuration_parses() -> None:
    config = load_run_config(Path("configs/examples/breast_cancer_logreg.yaml"))
    assert config.dataset.name == "breast_cancer"
    assert config.split.n_splits == 5
    assert len(config.fingerprint) == 64


def test_fingerprint_is_canonical_and_ignores_output_location(example_config: RunConfig) -> None:
    data = example_config.model_dump(mode="json")
    reordered = {key: data[key] for key in reversed(data)}
    reordered["output"] = {"directory": "somewhere-else"}
    assert RunConfig.model_validate(reordered).fingerprint == example_config.fingerprint


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": {"strategy": "stratified_kfold", "n_splits": 1}}, "greater than or equal"),
        ({"metrics": []}, "at least one metric"),
        ({"unexpected": True}, "Extra inputs are not permitted"),
        (
            {"model": {"name": "logistic_regression", "parameters": {"random_state": 7}}},
            "controlled by the top-level seed",
        ),
    ],
)
def test_malformed_configuration_fails_clearly(
    example_config: RunConfig, change: dict[str, object], message: str
) -> None:
    data = example_config.model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(data)
