from pathlib import Path

import pytest
from pydantic import ValidationError

from benchforge.core.config import FileDatasetConfig, SearchConfig, TaskType, load_search_config
from benchforge.search.registry import default_search_space_registry


def test_search_configuration_parses() -> None:
    config = load_search_config("configs/examples/breast_cancer_search.yaml")
    assert config.outer_split.n_splits == 2
    assert config.inner_split.n_splits == 2
    assert config.budget.trials_per_outer_fold == 2
    assert sum(model.mode == "search" for model in config.models) == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"models": []}, "at least one search candidate"),
        ({"budget": {"trials_per_outer_fold": 0}}, "greater than or equal to 1"),
        ({"primary_metric": "accuracy"}, "primary_metric must also be present"),
    ],
)
def test_invalid_search_configuration_fails(change: dict[str, object], message: str) -> None:
    data = load_search_config("configs/examples/breast_cancer_search.yaml").model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError, match=message):
        SearchConfig.model_validate(data)


def test_a_fixed_only_search_is_rejected() -> None:
    data = load_search_config("configs/examples/breast_cancer_search.yaml").model_dump(mode="json")
    data["models"] = [{"name": "dummy_classifier", "mode": "fixed"}]
    with pytest.raises(ValidationError, match="at least one model must use mode='search'"):
        SearchConfig.model_validate(data)


def test_default_spaces_cover_searchable_families() -> None:
    assert set(default_search_space_registry.names) == {
        "logistic_regression",
        "random_forest_classifier",
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
    }
    with pytest.raises(ValueError, match="unsupported search family"):
        default_search_space_registry.resolve("dummy_classifier")


def test_search_fingerprint_is_stable_and_excludes_output() -> None:
    config = load_search_config(Path("configs/examples/breast_cancer_search.yaml"))
    names = {model.name for model in config.models if model.mode == "search"}
    spaces = default_search_space_registry.identity_for(names)
    identity = "sklearn:breast_cancer:v1"
    first = config.fingerprint_for_dataset(identity, spaces)
    moved = config.model_copy(
        update={"output": config.output.model_copy(update={"directory": Path("elsewhere")})}
    )
    assert moved.fingerprint_for_dataset(identity, spaces) == first
    changed_budget = config.model_copy(
        update={
            "budget": config.budget.model_copy(
                update={"trials_per_outer_fold": config.budget.trials_per_outer_fold + 1}
            )
        }
    )
    assert changed_budget.fingerprint_for_dataset(identity, spaces) != first
    changed_spaces = {**spaces, "schema_revision": 2}
    assert config.fingerprint_for_dataset(identity, changed_spaces) != first


def test_file_relocation_with_identical_content_preserves_search_identity(
    tmp_path: Path,
) -> None:
    source = Path("configs/examples/data/mixed_customers.csv")
    relocated = tmp_path / "renamed.csv"
    relocated.write_bytes(source.read_bytes())
    config = load_search_config("configs/examples/breast_cancer_search.yaml")
    names = {model.name for model in config.models if model.mode == "search"}
    spaces = default_search_space_registry.identity_for(names)
    first = config.model_copy(
        update={
            "dataset": FileDatasetConfig(
                source="csv",
                path=source,
                target_column="subscribed",
                id_columns=("customer_id",),
                task=TaskType.BINARY_CLASSIFICATION,
            )
        }
    )
    second = first.model_copy(
        update={"dataset": first.dataset.model_copy(update={"path": relocated})}
    )
    identity = "csv:sha256:same-content"
    assert first.fingerprint_for_dataset(identity, spaces) == second.fingerprint_for_dataset(
        identity, spaces
    )
