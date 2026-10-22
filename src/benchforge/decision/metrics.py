"""Probability diagnostics and exact, deterministic threshold selection semantics."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Literal

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from benchforge.core.config import MetricName, TaskType
from benchforge.decision.config import ThresholdConfig
from benchforge.decision.results import (
    Aggregate,
    CandidateResult,
    Confusion,
    MetricValue,
    OuterEvaluation,
    PolicyEvaluation,
    ProbabilityQuality,
    ReliabilityBin,
    ThresholdSelection,
    ThresholdTrial,
)
from benchforge.evaluation.metrics import aggregate_fold_metrics, compute_metrics


def direction(name: str) -> Literal["maximize", "minimize"]:
    return (
        "minimize"
        if name
        in {
            "selected.misclassification_cost",
            "reference.misclassification_cost",
            "probability.brier_score",
            "probability.log_loss",
            "probability.ece",
        }
        else "maximize"
    )


def validated_arrays(
    truth: Sequence[int], probabilities: Sequence[float]
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
    y = np.asarray(truth)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("truth/probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("truth must use encoded binary labels 0 and 1")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y.astype(np.int64), p


def confusion(truth: Sequence[int], prediction: Sequence[int]) -> Confusion:
    y = np.asarray(truth)
    pred = np.asarray(prediction)
    if y.shape != pred.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("predictions must be aligned binary labels")
    return Confusion(
        tn=int(np.sum((y == 0) & (pred == 0))),
        fp=int(np.sum((y == 0) & (pred == 1))),
        fn=int(np.sum((y == 1) & (pred == 0))),
        tp=int(np.sum((y == 1) & (pred == 1))),
    )


def exact_objective(counts: Confusion, policy: ThresholdConfig) -> Fraction:
    """Compare rational counts/costs exactly, including decimal-configured costs."""
    c = counts
    if policy.objective == "f1":
        denominator = 2 * c.tp + c.fp + c.fn
        return Fraction(2 * c.tp, denominator) if denominator else Fraction(0)
    if policy.objective == "balanced_accuracy":
        if not (c.tp + c.fn and c.tn + c.fp):
            raise ValueError("balanced accuracy requires both truth classes")
        return (Fraction(c.tp, c.tp + c.fn) + Fraction(c.tn, c.tn + c.fp)) / 2
    n = c.tp + c.tn + c.fp + c.fn
    return (
        Fraction(str(policy.false_positive_cost)) * c.fp
        + Fraction(str(policy.false_negative_cost)) * c.fn
    ) / n


def select_threshold(
    truth: Sequence[int], probabilities: Sequence[float], policy: ThresholdConfig
) -> ThresholdSelection:
    """Select solely from supplied inner evidence; equality uses p >= threshold."""
    y, p = validated_arrays(truth, probabilities)
    grid = policy.candidates if policy.policy == "optimize" else (policy.value,)
    trials = []
    exact_values = []
    for threshold in grid:
        counts = confusion(y.tolist(), (p >= threshold).astype(int).tolist())
        value = exact_objective(counts, policy)
        exact_values.append(value)
        trials.append(
            ThresholdTrial(threshold=threshold, confusion=counts, objective_value=float(value))
        )
    sign = 1 if policy.objective == "misclassification_cost" else -1
    winner = min(
        range(len(grid)),
        key=lambda i: (
            sign * exact_values[i],
            abs(Decimal(str(grid[i])) - Decimal("0.5")),
            grid[i],
        ),
    )
    return ThresholdSelection(
        threshold=grid[winner],
        objective_value=trials[winner].objective_value,
        trials=tuple(trials),
    )


def evaluate_policy(
    truth: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    costs: ThresholdConfig,
) -> PolicyEvaluation:
    y, p = validated_arrays(truth, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    predicted = (p >= threshold).astype(np.int64)
    c = confusion(y.tolist(), predicted.tolist())
    metrics = compute_metrics(
        (MetricName.ACCURACY, MetricName.BALANCED_ACCURACY, MetricName.F1),
        y,
        predicted,
        None,
        TaskType.BINARY_CLASSIFICATION,
    )
    metrics.update(
        precision=c.tp / (c.tp + c.fp) if c.tp + c.fp else 0.0,
        recall=c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0,
        misclassification_cost=float(
            exact_objective(c, costs.model_copy(update={"objective": "misclassification_cost"}))
        ),
    )
    return PolicyEvaluation(
        threshold=threshold,
        confusion=c,
        metrics=tuple(MetricValue(name=k, value=v) for k, v in sorted(metrics.items())),
    )


def probability_quality(
    truth: Sequence[int], probabilities: Sequence[float], n_bins: int
) -> ProbabilityQuality:
    """Uniform bins [lower,upper), last inclusive; ECE is descriptive only."""
    y, p = validated_arrays(truth, probabilities)
    if not 2 <= n_bins <= 100:
        raise ValueError("reliability bin count must be in [2, 100]")
    # searchsorted fixes floating boundary semantics to the exact persisted edges.
    edges = np.linspace(0, 1, n_bins + 1)
    positions = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        included = positions == i
        count = int(included.sum())
        mean = float(np.mean(p[included])) if count else None
        fraction = float(np.mean(y[included])) if count else None
        if mean is not None and fraction is not None:
            ece += count / len(y) * abs(mean - fraction)
        bins.append(
            ReliabilityBin(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                count=count,
                mean_probability=mean,
                positive_fraction=fraction,
            )
        )
    # sklearn log_loss clips float64 predictions at machine epsilon, including 0/1.
    return ProbabilityQuality(
        brier_score=float(np.mean((p - y) ** 2)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        ece=ece,
        bins=tuple(bins),
    )


def fold_metrics(fold: OuterEvaluation) -> dict[str, float]:
    values = {
        f"{prefix}.{m.name}": m.value
        for prefix, policy in (("selected", fold.selected), ("reference", fold.reference))
        for m in policy.metrics
    }
    q = fold.probability_quality
    values.update(
        {
            f"probability.{k}": v
            for k, v in (
                ("brier_score", q.brier_score),
                ("log_loss", q.log_loss),
                ("roc_auc", q.roc_auc),
                ("ece", q.ece),
            )
            if v is not None
        }
    )
    return values


def aggregate(folds: tuple[OuterEvaluation, ...]) -> tuple[Aggregate, ...]:
    if not folds:
        return ()
    rows = [fold_metrics(f) for f in folds]
    # Costs can be very large finite values. Rescale before the existing kernel
    # so summation and squared deviations do not overflow unnecessarily.
    scales = {name: max(1.0, max(abs(row[name]) for row in rows)) for name in rows[0]}
    normalized = ({name: value / scales[name] for name, value in row.items()} for row in rows)
    return tuple(
        Aggregate(name=name, mean=m.mean * scales[name], std=m.std * scales[name])
        for name, m in sorted(aggregate_fold_metrics(normalized).items())
    )


def rank_candidates(
    candidates: tuple[CandidateResult, ...], primary: str
) -> tuple[CandidateResult, ...]:
    """Rank complete candidates only; exact ties share competition ranks."""
    scores = {
        c.identifier: next(m.mean for m in c.metrics if m.name == primary)
        for c in candidates
        if c.status == "success"
    }
    sign = -1 if direction(primary) == "maximize" else 1
    order = sorted(scores, key=lambda identifier: (sign * scores[identifier], identifier))
    ranks = {
        identifier: 1 + sum(sign * other < sign * scores[identifier] for other in scores.values())
        for identifier in order
    }
    return tuple(
        c.model_copy(update={"rank": ranks.get(c.identifier)})
        for c in sorted(candidates, key=lambda c: c.identifier)
    )
