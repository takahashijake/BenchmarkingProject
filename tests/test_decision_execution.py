"""Scientific boundaries, calibration nesting, determinism and failure isolation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import confusion_matrix, f1_score

from benchforge.data.registry import default_dataset_registry
from benchforge.decision import DecisionAnalysisError, DecisionConfig, run_decision_analysis
from benchforge.decision.planning import build_plan
from benchforge.decision.runner import evaluate_outer, predict_probabilities
from benchforge.decision.verification import validate_decision


@pytest.fixture
def decision_config():
    return DecisionConfig.model_validate(
        {
            "dataset": {"name": "breast_cancer"},
            "outer_split": {"n_splits": 2},
            "inner_split": {"n_splits": 2},
            "models": [
                {"name": "logistic_regression", "id": "logistic", "parameters": {"max_iter": 1000}},
                {"name": "dummy_classifier", "id": "dummy"},
            ],
            "threshold": {"policy": "optimize", "candidates": [0, 0.25, 0.5, 0.75, 1]},
            "calibration": {"method": "sigmoid", "n_splits": 2},
        }
    )


@pytest.mark.parametrize("method", ["none", "sigmoid"])
def test_determinism_shared_folds_and_independent_outer_reconstruction(decision_config, method):
    raw = decision_config.model_dump(mode="json")
    raw["calibration"]["method"] = method
    config = DecisionConfig.model_validate(raw)
    first = run_decision_analysis(config, persist=False)
    second = run_decision_analysis(config, persist=False)
    assert first == second
    validate_decision(config, first)
    fold_sets = []
    for candidate in first.candidates:
        sample_folds = []
        for fold, plan in zip(candidate.folds, first.plan.outer, strict=True):
            outer_train = set(plan.final.fold.train_indices)
            outer_valid = set(plan.final.fold.validation_indices)
            inner_coverage = set()
            for boundary, inner in zip(plan.inner, fold.inner, strict=True):
                train, valid = (
                    set(boundary.fold.train_indices),
                    set(boundary.fold.validation_indices),
                )
                assert train | valid == outer_train
                assert not train & valid and not (train | valid) & outer_valid
                inner_coverage |= valid
                assert {p.sample_index for p in inner.probabilities} == valid
            assert inner_coverage == outer_train
            y = [p.truth for p in fold.predictions]
            pred = [p.prediction for p in fold.predictions]
            ref = [p.reference_prediction for p in fold.predictions]
            probs = [p.probability for p in fold.predictions]
            assert all(0 <= p <= 1 and np.isfinite(p) for p in probs)
            assert pred == [int(p >= fold.selection.threshold) for p in probs]
            assert ref == [int(p >= 0.5) for p in probs]
            assert confusion_matrix(y, pred).ravel().tolist() == list(
                fold.selected.confusion.model_dump().values()
            )
            assert f1_score(y, ref) == next(
                m.value for m in fold.reference.metrics if m.name == "f1"
            )
            assert f1_score(y, pred) == next(
                m.value for m in fold.selected.metrics if m.name == "f1"
            )
            assert [p.sample_index for p in fold.predictions] == list(outer_valid) or (
                set(p.sample_index for p in fold.predictions) == outer_valid
            )
            # Reconstruct the entire threshold choice without invoking the optimizer.
            inner = [p for f in fold.inner for p in f.probabilities]
            values = {
                t: f1_score([p.truth for p in inner], [p.probability >= t for p in inner])
                for t in config.threshold.candidates
            }
            best = max(values.values())
            ties = [t for t, score in values.items() if score == best]
            assert fold.selection.threshold == min(ties, key=lambda t: (abs(t - 0.5), t))
            sample_folds.extend((p.sample_index, p.fold_id) for p in fold.predictions)
        assert sorted(p for p, f in sample_folds) == list(range(first.dataset.row_count))
        fold_sets.append(sorted(sample_folds))
    assert fold_sets[0] == fold_sets[1]
    assert first.dataset.target_labels == ("malignant", "benign")


class AuditPreprocessor(TransformerMixin, BaseEstimator):
    events = []
    permitted = set()

    def fit(self, x, y=None):
        indices = set(x.index)
        assert indices <= self.permitted
        self.events.append(indices)
        return self

    def transform(self, x):
        return x


class AuditClassifier(ClassifierMixin, BaseEstimator):
    events = []
    permitted = set()

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, x, y):
        self.seen_ = set(x.index)
        assert self.seen_ <= self.permitted
        self.events.append(self.seen_)
        self.classes_ = np.array([0, 1])
        self.mean_ = float(x.iloc[:, 0].mean())
        return self

    def decision_function(self, x):
        assert not self.seen_ & set(x.index), "evaluation observations leaked into fitting"
        return (x.iloc[:, 0].to_numpy() - self.mean_) / 5

    def predict(self, x):
        return (self.decision_function(x) >= 0).astype(int)

    def predict_proba(self, x):
        margin = self.decision_function(x)
        p = 1 / (1 + np.exp(-margin))
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_sentinel_all_preprocessing_base_fits_and_calibration_are_training_only(
    monkeypatch,
    decision_config,
    method,
):
    from benchforge.execution import runner as kernel
    from benchforge.models.registry import default_model_registry

    dataset = default_dataset_registry.resolve("breast_cancer")
    if method == "isotonic":
        # Three nesting layers must still leave >=1000 pooled calibration rows.
        x = pd.concat([dataset.features] * 8, ignore_index=True)
        y = pd.concat([dataset.target] * 8, ignore_index=True)
        dataset = replace(dataset, features=x, target=y)
    config = DecisionConfig.model_validate(
        {
            **decision_config.model_dump(mode="json"),
            "calibration": {"method": method, "n_splits": 2},
        }
    )
    plan = build_plan(config, tuple(int(y) for y in dataset.target)).outer[0]
    AuditClassifier.events, AuditPreprocessor.events = [], []
    monkeypatch.setattr(kernel, "build_preprocessor", lambda *a, **kw: AuditPreprocessor())
    monkeypatch.setattr(
        default_model_registry,
        "create",
        lambda model, seed, task: AuditClassifier(random_state=seed),
    )
    for boundary in (*plan.inner, plan.final):
        AuditClassifier.permitted = set(boundary.fold.train_indices)
        AuditPreprocessor.permitted = set(boundary.fold.train_indices)
        start = len(AuditClassifier.events)
        p = predict_probabilities(config, dataset, config.models[0], boundary, 123)
        assert len(p) == len(boundary.fold.validation_indices)
        assert all(np.isfinite(p)) and all(0 <= value <= 1 for value in p)
        trained = AuditClassifier.events[start:]
        expected = [set(boundary.fold.train_indices)]
        if boundary.calibration:
            expected = [set(f.train_indices) for f in boundary.calibration.folds] + expected
            assert all(
                not set(f.validation_indices) & set(f.train_indices)
                for f in boundary.calibration.folds
            )
        assert trained == expected
        assert AuditPreprocessor.events[start:] == expected


@pytest.mark.parametrize("method", ["none", "sigmoid"])
def test_outer_outcome_and_feature_sentinel_cannot_change_inner_selection(decision_config, method):
    config = DecisionConfig.model_validate(
        {
            **decision_config.model_dump(mode="json"),
            "calibration": {"method": method, "n_splits": 2},
        }
    )
    dataset = default_dataset_registry.resolve("breast_cancer")
    plan = build_plan(config, tuple(int(y) for y in dataset.target)).outer[0]
    normal = evaluate_outer(config, dataset, config.models[0], plan)
    x, y = dataset.features.copy(), dataset.target.copy()
    valid = list(plan.final.fold.validation_indices)
    x.iloc[valid] = x.iloc[valid] * 1000 + 10000
    y.iloc[valid] = 1 - y.iloc[valid]
    altered = evaluate_outer(config, replace(dataset, features=x, target=y), config.models[0], plan)
    assert normal.inner == altered.inner
    assert normal.selection == altered.selection
    assert normal.predictions != altered.predictions


def test_probability_orientation_uses_class_one_even_with_reversed_columns(
    monkeypatch, decision_config
):
    from benchforge.decision import runner

    class Reversed:
        classes_ = np.array([1, 0])

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.tile([0.8, 0.2], (len(x), 1))

    config = DecisionConfig.model_validate(
        {**decision_config.model_dump(mode="json"), "calibration": {"method": "none"}}
    )
    dataset = default_dataset_registry.resolve("breast_cancer")
    boundary = build_plan(config, tuple(int(y) for y in dataset.target)).outer[0].final
    monkeypatch.setattr(runner, "build_pipeline", lambda *a: Reversed())
    assert set(predict_probabilities(config, dataset, config.models[0], boundary, 1)) == {0.8}


def test_margin_only_estimator_requires_calibration(monkeypatch, decision_config):
    from sklearn.svm import LinearSVC

    from benchforge.models.registry import default_model_registry

    monkeypatch.setattr(default_model_registry, "create", lambda *a: LinearSVC(random_state=0))
    raw = decision_config.model_dump(mode="json")
    raw["calibration"]["method"] = "none"
    with pytest.raises(ValueError, match="genuine predict_proba"):
        run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    raw["calibration"]["method"] = "sigmoid"
    result = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    assert result.status == "success"
    assert all(
        0 <= p.probability <= 1 for c in result.candidates for f in c.folds for p in f.predictions
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_insufficient_class_counts_fail_before_training(monkeypatch, decision_config, count):
    from benchforge.decision import runner

    dataset = default_dataset_registry.resolve("breast_cancer")
    indices = list(dataset.target[dataset.target == 0].index[:count]) + list(
        dataset.target[dataset.target == 1].index[:count]
    )
    dataset = replace(
        dataset,
        features=dataset.features.iloc[indices].reset_index(drop=True),
        target=dataset.target.iloc[indices].reset_index(drop=True),
    )
    monkeypatch.setattr(default_dataset_registry, "resolve", lambda *a: dataset)
    monkeypatch.setattr(runner, "evaluate_outer", lambda *a: pytest.fail("unexpected training"))
    with pytest.raises(ValueError, match="each target class|binary classification"):
        run_decision_analysis(decision_config, persist=False)


def test_single_class_and_isotonic_small_sample_fail_before_training(monkeypatch, decision_config):
    from benchforge.decision import runner

    monkeypatch.setattr(runner, "evaluate_outer", lambda *a: pytest.fail("unexpected training"))
    raw = decision_config.model_dump(mode="json")
    raw["calibration"]["method"] = "isotonic"
    with pytest.raises(ValueError, match="isotonic requires at least 1000"):
        run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    with pytest.raises(ValueError, match="exactly two"):
        build_plan(decision_config, (0,) * 100)


def test_candidate_failure_isolation_and_partial_candidates_never_rank(
    monkeypatch, decision_config
):
    from benchforge.decision import runner

    original = runner.evaluate_outer

    def fail(config, dataset, model, plan):
        if model.id == "logistic" and plan.final.fold.fold_id == 1:
            raise ValueError("deliberate scientific fold failure")
        return original(config, dataset, model, plan)

    monkeypatch.setattr(runner, "evaluate_outer", fail)
    result = run_decision_analysis(decision_config, persist=False)
    candidates = {c.identifier: c for c in result.candidates}
    assert candidates["dummy"].status == "success" and candidates["dummy"].rank == 1
    partial = candidates["logistic"]
    assert partial.status == "partial" and partial.rank is None and len(partial.folds) == 1
    assert partial.failure.fold_id == 1
    assert result.status == "partial"
    validate_decision(decision_config, result)


def test_all_failed_evidence_is_available(monkeypatch, decision_config):
    from benchforge.decision import runner

    def fail(*args):
        raise ValueError("deliberate failure")

    monkeypatch.setattr(runner, "evaluate_outer", fail)
    with pytest.raises(DecisionAnalysisError) as exc:
        run_decision_analysis(decision_config, persist=False)
    result = exc.value.result
    assert result.status == "failed" and all(c.rank is None for c in result.candidates)
    validate_decision(decision_config, result)


@pytest.mark.parametrize(
    "name", ["ridge_regressor", "unsupported_classifier", "example.scaled_logistic"]
)
def test_unsupported_models_fail_before_data_loading(monkeypatch, decision_config, name):
    monkeypatch.setattr(default_dataset_registry, "resolve", lambda *a: pytest.fail("data loaded"))
    raw = decision_config.model_dump(mode="json")
    raw["models"] = [{"name": name}]
    with pytest.raises(ValueError, match="unsupported model|supports tasks"):
        run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)


def test_result_is_deeply_immutable_and_model_copy_is_revalidated(decision_config):
    from pydantic import ValidationError

    result = run_decision_analysis(decision_config, persist=False)
    with pytest.raises(ValidationError):
        result.candidates[0].folds[0].predictions[0].probability = 1.0
    with pytest.raises(TypeError):
        result.plan.outer[0].final.fold.train_indices[0] = 100
    with pytest.raises(ValidationError):
        run_decision_analysis(decision_config.model_copy(update={"seed": -1}), persist=False)


def test_fixed_policy_preserves_matched_reference(decision_config):
    raw = decision_config.model_dump(mode="json")
    raw["threshold"] = {"policy": "fixed", "value": 0.5}
    result = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    for candidate in result.candidates:
        for fold in candidate.folds:
            assert fold.selected == fold.reference
            assert all(p.prediction == p.reference_prediction for p in fold.predictions)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("logistic_regression", {"max_iter": 1000}),
        ("dummy_classifier", {}),
        ("random_forest_classifier", {"n_estimators": 5, "max_depth": 3}),
        ("extra_trees_classifier", {"n_estimators": 5, "max_depth": 3}),
        ("hist_gradient_boosting_classifier", {"max_iter": 5}),
    ],
)
@pytest.mark.parametrize("method", ["none", "sigmoid"])
def test_all_builtin_binary_estimators_use_real_pipeline(decision_config, name, parameters, method):
    raw = decision_config.model_dump(mode="json")
    raw["models"] = [{"name": name, "parameters": parameters}]
    raw["calibration"]["method"] = method
    result = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    assert result.status == "success" and result.candidates[0].rank == 1


def test_competition_ranks_preserve_exact_candidate_ties(decision_config):
    raw = decision_config.model_dump(mode="json")
    raw["models"] = [
        {"name": "dummy_classifier", "id": "a"},
        {"name": "dummy_classifier", "id": "b"},
    ]
    result = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    assert [c.rank for c in result.candidates] == [1, 1]


def test_canonical_candidate_and_grid_order_and_global_rng_are_unchanged(decision_config):
    raw = decision_config.model_dump(mode="json")
    first = run_decision_analysis(decision_config, persist=False)
    raw["models"].reverse()
    raw["threshold"]["candidates"].reverse()
    raw["output"] = {"directory": "a-different-path"}
    before = np.random.get_state()
    second = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    after = np.random.get_state()
    assert first == second
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_actual_invalid_estimator_isolated_without_rank(decision_config):
    raw = decision_config.model_dump(mode="json")
    raw["models"][0]["parameters"]["C"] = -1
    result = run_decision_analysis(DecisionConfig.model_validate(raw), persist=False)
    failed = next(c for c in result.candidates if c.identifier == "logistic")
    assert failed.status == "failed" and failed.rank is None and not failed.metrics
    assert failed.failure.fold_id == 0 and result.status == "partial"


def test_large_costs_have_finite_outer_aggregates(decision_config):
    raw = decision_config.model_dump(mode="json")
    raw["models"] = [{"name": "dummy_classifier", "parameters": {"strategy": "uniform"}}]
    raw["threshold"].update(
        false_positive_cost=1e308, false_negative_cost=1e308, objective="misclassification_cost"
    )
    raw["primary_metric"] = "selected.misclassification_cost"
    config = DecisionConfig.model_validate(raw)
    result = run_decision_analysis(config, persist=False)
    assert all(np.isfinite(m.mean) and np.isfinite(m.std) for m in result.candidates[0].metrics)
    validate_decision(config, result)


def test_real_isotonic_full_workflow_and_evidence(monkeypatch, decision_config):
    dataset = default_dataset_registry.resolve("breast_cancer")
    dataset = replace(
        dataset,
        features=pd.concat([dataset.features] * 8, ignore_index=True),
        target=pd.concat([dataset.target] * 8, ignore_index=True),
    )
    monkeypatch.setattr(default_dataset_registry, "resolve", lambda *a: dataset)
    raw = decision_config.model_dump(mode="json")
    raw["calibration"]["method"] = "isotonic"
    raw["models"] = [{"name": "dummy_classifier"}]
    config = DecisionConfig.model_validate(raw)
    result = run_decision_analysis(config, persist=False)
    assert result.status == "success"
    validate_decision(config, result)
    assert all(p.final.calibration is not None for p in result.plan.outer)
