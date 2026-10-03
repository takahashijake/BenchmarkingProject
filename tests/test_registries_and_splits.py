import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from benchforge.core.config import ModelConfig, SplitConfig, TaskType
from benchforge.data import default_dataset_registry
from benchforge.models import default_model_registry
from benchforge.splits import build_stratified_folds


def test_dataset_registry_resolves_offline_dataset() -> None:
    dataset = default_dataset_registry.resolve("breast_cancer")
    assert dataset.identity == "sklearn:breast_cancer:v1"
    assert dataset.task == TaskType.BINARY_CLASSIFICATION
    assert dataset.features.shape == (569, 30)
    assert len(dataset.feature_names) == 30


def test_split_generation_is_deterministic() -> None:
    target = default_dataset_registry.resolve("breast_cancer").target
    config = SplitConfig(n_splits=4)
    first = build_stratified_folds(target, config, seed=19)
    second = build_stratified_folds(target, config, seed=19)
    assert all(
        np.array_equal(left.validation_indices, right.validation_indices)
        for left, right in zip(first, second, strict=True)
    )
    validation_indices = np.concatenate([fold.validation_indices for fold in first])
    assert set(validation_indices) == set(range(len(target)))


def test_model_registry_resolves_supported_models() -> None:
    logistic = default_model_registry.create(ModelConfig(name="logistic_regression"), seed=3)
    forest = default_model_registry.create(ModelConfig(name="random_forest_classifier"), seed=3)
    assert isinstance(logistic, LogisticRegression)
    assert isinstance(forest, RandomForestClassifier)
    assert logistic.random_state == forest.random_state == 3
    assert isinstance(
        default_model_registry.create(ModelConfig(name="extra_trees_classifier"), seed=3),
        ExtraTreesClassifier,
    )
    assert isinstance(
        default_model_registry.create(
            ModelConfig(name="hist_gradient_boosting_classifier"), seed=3
        ),
        HistGradientBoostingClassifier,
    )
    assert isinstance(
        default_model_registry.create(ModelConfig(name="dummy_classifier"), seed=3),
        DummyClassifier,
    )
    assert default_model_registry.capabilities("hist_gradient_boosting_classifier").requires_dense


def test_model_registry_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="unsupported model 'made_up'"):
        default_model_registry.create(ModelConfig(name="made_up"), seed=3)
