"""Independent numeric checks for decision contract 1."""

import math
from decimal import Decimal

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.metrics import balanced_accuracy_score, f1_score

from benchforge.decision.config import DecisionConfig, ThresholdConfig
from benchforge.decision.metrics import evaluate_policy, probability_quality, select_threshold


@pytest.mark.parametrize("objective", ["f1", "balanced_accuracy", "misclassification_cost"])
def test_threshold_matches_independent_enumeration(objective):
    y = np.array([0] * 17 + [1] * 3)
    p = np.array(
        [
            0.01,
            0.04,
            0.1,
            0.13,
            0.19,
            0.2,
            0.21,
            0.22,
            0.24,
            0.25,
            0.27,
            0.28,
            0.31,
            0.33,
            0.6,
            0.8,
            0.95,
            0.24,
            0.63,
            0.99,
        ]
    )
    config = ThresholdConfig(
        policy="optimize", objective=objective, false_positive_cost=2.5, false_negative_cost=9.0
    )
    values = {}
    for t in config.candidates:
        pred = p >= t
        if objective == "f1":
            values[t] = f1_score(y, pred)
        elif objective == "balanced_accuracy":
            values[t] = balanced_accuracy_score(y, pred)
        else:
            values[t] = (2.5 * np.sum((y == 0) & pred) + 9 * np.sum((y == 1) & ~pred)) / len(y)
    optimum = (
        min(values.values()) if objective == "misclassification_cost" else max(values.values())
    )
    tied = [t for t in values if values[t] == optimum]
    expected = min(tied, key=lambda t: (abs(Decimal(str(t)) - Decimal("0.5")), t))
    selected = select_threshold(y, p, config)
    assert selected.threshold == expected
    assert selected.objective_value == pytest.approx(optimum)
    assert len(selected.trials) == len(config.candidates)


def test_exact_decimal_ties_choose_smaller_equidistant_threshold():
    c = ThresholdConfig(
        policy="optimize",
        candidates=(0.6, 0.4),
        objective="misclassification_cost",
        false_positive_cost=0.0,
        false_negative_cost=0.0,
    )
    assert select_threshold([0, 1], [0.1, 0.9], c).threshold == 0.4
    c = c.model_copy(update={"candidates": (0.4, 0.5, 0.6)})
    assert select_threshold([0, 1], [0.1, 0.9], c).threshold == 0.5


@pytest.mark.parametrize("objective", ["f1", "balanced_accuracy", "misclassification_cost"])
def test_boundary_and_degenerate_predictions(objective):
    c = ThresholdConfig(policy="optimize", objective=objective, candidates=(0.0, 0.5, 1.0))
    selection = select_threshold([0, 1, 0, 1], [0.1] * 4, c)
    assert selection.trials[0].confusion.model_dump() == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}
    assert selection.trials[-1].confusion.model_dump() == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}
    boundary = evaluate_policy([0, 1], [0.0, 1.0], 1.0, c)
    assert boundary.confusion.model_dump() == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert evaluate_policy([0, 1], [0.0, 1.0], 0.0, c).confusion.fp == 1


def test_fixed_policy_and_asymmetric_cost_are_independent_of_objective():
    c = ThresholdConfig(value=0.75, false_positive_cost=3.0, false_negative_cost=8.0)
    result = evaluate_policy([0, 0, 1, 1], [0.2, 0.8, 0.5, 0.9], c.value, c)
    assert result.confusion.model_dump() == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    values = {m.name: m.value for m in result.metrics}
    assert values == {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "f1": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "misclassification_cost": 2.75,
    }
    selection = select_threshold([0, 1], [0.1, 0.9], c)
    assert selection.threshold == 0.75 and len(selection.trials) == 1


def test_extreme_finite_costs_do_not_overflow_intermediate_sums():
    c = ThresholdConfig(false_positive_cost=1e308, false_negative_cost=1e308)
    v = evaluate_policy([0, 0, 1, 1], [1, 1, 0, 0], 0.5, c)
    assert next(m.value for m in v.metrics if m.name == "misclassification_cost") == 1e308


def test_probability_quality_and_uniform_bins_independently():
    y = [0, 1, 0, 1, 1]
    p = [0.0, 0.25, 0.5, 0.75, 1.0]
    q = probability_quality(y, p, 4)
    assert q.brier_score == pytest.approx(sum((a - b) ** 2 for a, b in zip(y, p, strict=True)) / 5)
    eps = np.finfo(np.float64).eps
    clipped = [min(1 - eps, max(eps, x)) for x in p]
    expected_loss = -sum(
        a * math.log(b) + (1 - a) * math.log(1 - b) for a, b in zip(y, clipped, strict=True)
    ) / len(y)
    assert q.log_loss == pytest.approx(expected_loss)
    assert [b.count for b in q.bins] == [1, 1, 1, 2]
    assert [b.mean_probability for b in q.bins] == [0, 0.25, 0.5, 0.875]
    assert [b.positive_fraction for b in q.bins] == [0, 1, 0, 1]
    assert q.ece == pytest.approx((0.75 + 0.5 + 2 * 0.125) / 5)
    # Independently count positive-negative pair orderings, ties worth one half.
    positive = [p[i] for i, v in enumerate(y) if v == 1]
    negative = [p[i] for i, v in enumerate(y) if v == 0]
    expected_auc = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / 6
    assert q.roc_auc == pytest.approx(expected_auc)


def test_empty_bins_and_undefined_auc_are_explicit():
    q = probability_quality([0, 0], [0.0, 1.0], 4)
    assert q.roc_auc is None
    assert [b.count for b in q.bins] == [1, 0, 0, 1]
    assert q.bins[1].mean_probability is None
    assert q.bins[1].positive_fraction is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_fail(bad):
    with pytest.raises(ValueError, match="probabilities"):
        probability_quality([0, 1], [0.1, bad], 10)
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.1, bad], ThresholdConfig())


@pytest.mark.parametrize(
    "changes",
    [
        {"candidates": ()},
        {"candidates": (0.5, 0.5)},
        {"candidates": (-0.1, 0.5)},
        {"candidates": (float("nan"),)},
        {"candidates": (float("inf"),)},
        {"false_positive_cost": -1.0},
        {"false_negative_cost": float("inf")},
        {"value": 1.1},
        {"value": float("nan")},
        {"objective": "accuracy"},
        {"policy": "optimize", "value": 0.7},
    ],
)
def test_invalid_threshold_contracts(changes):
    with pytest.raises(ValidationError):
        ThresholdConfig.model_validate(changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "regression"},
        {"task": "multiclass_classification"},
        {"models": []},
        {"schema_version": 2},
        {"workers": 2},
        {"providers": []},
        {"checkpoint_dir": "workspace"},
        {"budget": {"trials": 2}},
        {"outer_split": {"strategy": "kfold"}},
        {"inner_split": {"n_splits": 1}},
        {"calibration": {"method": "temperature"}},
        {"calibration": {"method": "isotonic", "isotonic_min_samples": 2}},
        {"reliability_bins": 1},
        {"seed": -1},
        {"models": [{"name": "dummy_classifier", "parameters": {"n_jobs": -1}}]},
        {"models": [{"name": "dummy_classifier"}, {"name": "dummy_classifier"}]},
    ],
)
def test_unsupported_configuration_is_rejected(changes):
    raw = {"dataset": {"name": "breast_cancer"}, "models": [{"name": "dummy_classifier"}]}
    with pytest.raises(ValidationError):
        DecisionConfig.model_validate({**raw, **changes})
