from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from benchforge.core.config import MetricName


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    std: float


@dataclass(frozen=True)
class MetricSpec:
    name: MetricName
    direction: Literal["maximize", "minimize"]
    requires_score: bool = False


_METRIC_SPECS = {
    MetricName.ACCURACY: MetricSpec(MetricName.ACCURACY, "maximize"),
    MetricName.BALANCED_ACCURACY: MetricSpec(MetricName.BALANCED_ACCURACY, "maximize"),
    MetricName.F1: MetricSpec(MetricName.F1, "maximize"),
    MetricName.ROC_AUC: MetricSpec(MetricName.ROC_AUC, "maximize", requires_score=True),
}


def metric_spec(name: MetricName) -> MetricSpec:
    return _METRIC_SPECS[name]


def compute_metrics(
    requested: Sequence[MetricName],
    truth: NDArray[np.int64],
    prediction: NDArray[np.int64],
    score: NDArray[np.float64] | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in requested:
        if metric == MetricName.ACCURACY:
            value = accuracy_score(truth, prediction)
        elif metric == MetricName.BALANCED_ACCURACY:
            value = balanced_accuracy_score(truth, prediction)
        elif metric == MetricName.F1:
            value = f1_score(truth, prediction)
        elif metric == MetricName.ROC_AUC:
            if score is None:
                raise ValueError("metric 'roc_auc' requires predict_proba or decision_function")
            value = roc_auc_score(truth, score)
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
