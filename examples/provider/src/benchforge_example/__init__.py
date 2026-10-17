"""Independently installed deterministic provider using the public v1 contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from benchforge.core.config import TaskType
from benchforge.data import Dataset
from benchforge.models import ModelCapabilities, ModelSpec
from benchforge.providers import (
    Component,
    EstimatorDeclaration,
    MetricDeclaration,
    ProviderManifest,
    SpaceDeclaration,
)
from benchforge.providers.runtime import Provider
from benchforge.search.spaces import SearchSpace


def dataset():
    values = np.arange(60, dtype=float)
    features = pd.DataFrame({"value": values % 7, "group": (values % 3).astype(str)})
    return Dataset(
        identity="example:toy:v1",
        task=TaskType.BINARY_CLASSIFICATION,
        features=features,
        target=pd.Series((values.astype(int) % 2), name="target"),
        feature_names=("value", "group"),
        numeric_feature_names=("value",),
        categorical_feature_names=("group",),
        target_name="target",
        missing_values={"value": 0, "group": 0},
        target_labels=("negative", "positive"),
    )


def estimator(parameters, seed):
    return LogisticRegression(random_state=seed, **parameters)


def agreement(truth, prediction, score):
    return float(np.mean(truth == prediction))


def suggest(trial, fixed):
    return {} if "C" in fixed else {"C": trial.suggest_float("C", 0.1, 2.0)}


def provider():
    tasks = ("binary_classification", "multiclass_classification")
    return Provider(
        manifest=ProviderManifest(
            id="example",
            version="1.0.0",
            api_version=1,
            datasets=(Component(id="example.toy", tasks=("binary_classification",)),),
            estimators=(
                EstimatorDeclaration(id="example.logistic", tasks=tasks, requires_dense=True),
            ),
            metrics=(
                MetricDeclaration(
                    id="example.agreement",
                    tasks=tasks,
                    direction="maximize",
                    display_name="Example agreement",
                    semantics_version="1.0.0",
                ),
            ),
            search_spaces=(
                SpaceDeclaration(
                    id="example.logistic",
                    version=1,
                    definition_json='{"C":{"high":2.0,"low":0.1,"type":"float"}}',
                ),
            ),
        ),
        datasets={"example.toy": dataset},
        estimators={
            "example.logistic": ModelSpec(
                estimator,
                frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}),
                ModelCapabilities(requires_dense=True),
            )
        },
        metrics={"example.agreement": agreement},
        search_spaces={
            "example.logistic": SearchSpace(
                name="example.logistic",
                version=1,
                definition={"C": {"type": "float", "low": 0.1, "high": 2.0}},
                suggest=suggest,
            )
        },
    )
