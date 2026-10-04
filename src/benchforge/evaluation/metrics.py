from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from benchforge.core.config import MetricName, TaskType


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    std: float


@dataclass(frozen=True)
class MetricSpec:
    name: MetricName
    supported_tasks: frozenset[TaskType]
    direction: Literal["maximize", "minimize"]
    requires_score: bool = False
    display_name: str = ""

    def supports_task(self, task: TaskType) -> bool:
        return task in self.supported_tasks


CLASSIFICATION_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)


_METRIC_SPECS: dict[MetricName, MetricSpec] = {
    MetricName.ACCURACY: MetricSpec(
        MetricName.ACCURACY, CLASSIFICATION_TASKS, "maximize", display_name="Accuracy"
    ),
    MetricName.BALANCED_ACCURACY: MetricSpec(
        MetricName.BALANCED_ACCURACY,
        CLASSIFICATION_TASKS,
        "maximize",
        display_name="Balanced accuracy",
    ),
    MetricName.F1: MetricSpec(
        MetricName.F1,
        frozenset({TaskType.BINARY_CLASSIFICATION}),
        "maximize",
        display_name="F1",
    ),
    MetricName.F1_MACRO: MetricSpec(
        MetricName.F1_MACRO,
        frozenset({TaskType.MULTICLASS_CLASSIFICATION}),
        "maximize",
        display_name="Macro F1",
    ),
    MetricName.ROC_AUC: MetricSpec(
        MetricName.ROC_AUC,
        frozenset({TaskType.BINARY_CLASSIFICATION}),
        "maximize",
        requires_score=True,
        display_name="ROC-AUC",
    ),
    MetricName.MEAN_ABSOLUTE_ERROR: MetricSpec(
        MetricName.MEAN_ABSOLUTE_ERROR,
        frozenset({TaskType.REGRESSION}),
        "minimize",
        display_name="MAE",
    ),
    MetricName.ROOT_MEAN_SQUARED_ERROR: MetricSpec(
        MetricName.ROOT_MEAN_SQUARED_ERROR,
        frozenset({TaskType.REGRESSION}),
        "minimize",
        display_name="RMSE",
    ),
    MetricName.R2: MetricSpec(
        MetricName.R2, frozenset({TaskType.REGRESSION}), "maximize", display_name="R²"
    ),
}


def metric_spec(name: MetricName) -> MetricSpec:
    return _METRIC_SPECS[name]


def compute_metrics(
    requested: Sequence[MetricName],
    truth: NDArray[np.int64] | NDArray[np.float64],
    prediction: NDArray[np.int64] | NDArray[np.float64],
    score: NDArray[np.float64] | None,
    task: TaskType | None = None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in requested:
        spec = metric_spec(metric)
        if task is not None and not spec.supports_task(task):
            raise ValueError(f"metric '{metric}' is incompatible with task '{task}'")
        if metric == MetricName.ACCURACY:
            value = accuracy_score(truth, prediction)
        elif metric == MetricName.BALANCED_ACCURACY:
            value = balanced_accuracy_score(truth, prediction)
        elif metric == MetricName.F1:
            value = f1_score(truth, prediction)
        elif metric == MetricName.F1_MACRO:
            value = f1_score(truth, prediction, average="macro")
        elif metric == MetricName.ROC_AUC:
            if score is None:
                raise ValueError("metric 'roc_auc' requires predict_proba or decision_function")
            value = roc_auc_score(truth, score)
        elif metric == MetricName.MEAN_ABSOLUTE_ERROR:
            value = mean_absolute_error(truth, prediction)
        elif metric == MetricName.ROOT_MEAN_SQUARED_ERROR:
            value = root_mean_squared_error(truth, prediction)
        elif metric == MetricName.R2:
            value = r2_score(truth, prediction)
        else:  # pragma: no cover - enum validation makes this defensive
            raise ValueError(f"unsupported metric '{metric}'")
        values[metric.value] = float(value)
    return values


def aggregate_fold_metrics(
    folds: Iterable[Mapping[str, float]],
) -> dict[str, AggregateMetric]:
    fold_list = list(folds)
    if not fold_list:
        raise ValueError("cannot aggregate zero folds")
    names = fold_list[0].keys()
    return {
        name: AggregateMetric(
            mean=float(np.mean([fold[name] for fold in fold_list])),
            std=float(np.std([fold[name] for fold in fold_list], ddof=0)),
        )
        for name in names
    }
