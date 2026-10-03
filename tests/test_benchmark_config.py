from pathlib import Path

import pytest
from pydantic import ValidationError

from benchforge.core.config import BenchmarkConfig, load_benchmark_config


def test_benchmark_configuration_validates() -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    assert len(config.models) == 5
    assert config.primary_metric.value == "roc_auc"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"models": []}, "at least one model is required"),
        ({"primary_metric": "f1"}, "primary_metric must also be present"),
    ],
)
def test_invalid_benchmark_configuration_fails(change: dict[str, object], message: str) -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    data = config.model_dump(mode="json")
    data.update(change)
    if change.get("primary_metric") == "f1":
        data["metrics"] = ["accuracy", "roc_auc"]
    with pytest.raises(ValidationError, match=message):
        BenchmarkConfig.model_validate(data)


def test_duplicate_model_configurations_fail() -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    data = config.model_dump(mode="json")
    data["models"] = [data["models"][0], data["models"][0]]
    with pytest.raises(ValidationError, match="duplicate identical model"):
        BenchmarkConfig.model_validate(data)


def test_suite_fingerprint_is_stable_and_model_order_independent() -> None:
    config = load_benchmark_config(Path("configs/examples/breast_cancer_suite.yaml"))
    data = config.model_dump(mode="json")
    data["models"] = list(reversed(data["models"]))
    reordered = BenchmarkConfig.model_validate(data)
    identity = "sklearn:breast_cancer:v1"
    assert config.fingerprint_for_dataset(identity) == reordered.fingerprint_for_dataset(identity)
    assert config.fingerprint_for_dataset(identity) != config.fingerprint_for_dataset("other-data")
