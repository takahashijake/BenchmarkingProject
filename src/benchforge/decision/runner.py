"""Serial nested evaluation over the established data, split and pipeline kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from benchforge.core.config import MetricName, ModelConfig, RunConfig
from benchforge.data.identity import dataset_digest
from benchforge.data.registry import Dataset, default_dataset_registry
from benchforge.decision.config import DecisionConfig
from benchforge.decision.metrics import (
    aggregate,
    direction,
    evaluate_policy,
    probability_quality,
    rank_candidates,
    select_threshold,
)
from benchforge.decision.planning import (
    SEMANTICS,
    build_plan,
    canonical,
    estimator_seed,
    fingerprint,
)
from benchforge.decision.results import (
    CandidateResult,
    DatasetEvidence,
    DecisionPrediction,
    DecisionResult,
    Failure,
    FitBoundary,
    InnerEvidence,
    OuterEvaluation,
    OuterPlan,
    ProbabilityObservation,
)
from benchforge.execution.runner import build_pipeline
from benchforge.models.registry import default_model_registry
from benchforge.robustness.planner import dependency_versions


class DecisionAnalysisError(RuntimeError):
    """All candidates failed; complete failure evidence is still available."""

    def __init__(self, result: DecisionResult) -> None:
        self.result = result
        super().__init__(f"all decision candidates failed; artifacts: {result.artifact_directory}")


def _run_config(config: DecisionConfig, model: ModelConfig, seed: int) -> RunConfig:
    return RunConfig(
        task=config.task,
        dataset=config.dataset,
        split=config.outer_split,
        preprocessing=config.preprocessing,
        model=model,
        metrics=(MetricName.F1,),
        seed=seed,
    )


def _model_identities(config: DecisionConfig) -> dict[str, Any]:
    identities = {}
    for model in sorted(config.models, key=lambda m: m.id or m.name):
        estimator = default_model_registry.create(model, config.seed, config.task)
        if config.calibration.method == "none" and not hasattr(estimator, "predict_proba"):
            raise ValueError(f"{model.name}: calibration none requires genuine predict_proba")
        if not (hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function")):
            raise ValueError(f"{model.name}: no supported probability/calibration response")
        identities[model.id or model.name] = {
            "class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            "resolved_parameters": estimator.get_params(deep=False),
            "registry": default_model_registry.identity_for({model.name}),
        }
    # Reject unserializable/nonfinite defaults before any fitting.
    canonical(identities)
    return identities


def predict_probabilities(
    config: DecisionConfig,
    dataset: Dataset,
    model: ModelConfig,
    boundary: FitBoundary,
    seed: int,
) -> tuple[float, ...]:
    """Fit only boundary.train, with preprocessing inside calibration cross-fitting."""
    pipeline = build_pipeline(_run_config(config, model, seed), dataset)
    fitted: Any = pipeline
    train = list(boundary.fold.train_indices)
    validation = list(boundary.fold.validation_indices)
    if boundary.calibration is not None:
        positions = {global_index: i for i, global_index in enumerate(train)}
        cv = [
            (
                np.asarray([positions[i] for i in f.train_indices], dtype=np.int64),
                np.asarray([positions[i] for i in f.validation_indices], dtype=np.int64),
            )
            for f in boundary.calibration.folds
        ]
        fitted = CalibratedClassifierCV(
            estimator=pipeline,
            method=config.calibration.method,
            cv=cv,
            ensemble=False,
            n_jobs=1,
        )
    fitted.fit(dataset.features.iloc[train], dataset.target.iloc[np.asarray(train, dtype=np.int64)])
    classes = np.asarray(fitted.classes_)
    if sorted(classes.tolist()) != [0, 1]:
        raise ValueError("decision probabilities require exactly the classes 0 and 1")
    positions_positive = np.flatnonzero(classes == 1)
    if len(positions_positive) != 1:
        raise ValueError("decision probabilities require a positive class labeled 1")
    raw = np.asarray(fitted.predict_proba(dataset.features.iloc[validation]), dtype=np.float64)
    if raw.shape != (len(validation), 2):
        raise ValueError("binary predict_proba must return two columns")
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("predict_proba returned nonfinite or out-of-range probabilities")
    if not np.allclose(raw.sum(axis=1), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("predict_proba rows must sum to one")
    return tuple(float(p) for p in raw[:, positions_positive[0]])


def evaluate_outer(
    config: DecisionConfig,
    dataset: Dataset,
    model: ModelConfig,
    plan: OuterPlan,
) -> OuterEvaluation:
    """Select on inner OOF, then predict untouched outer validation exactly once."""
    outer_id = plan.final.fold.fold_id
    identifier = model.id or model.name
    inner_evidence = []
    for boundary in plan.inner:
        fold = boundary.fold
        seed = estimator_seed(config, identifier, outer_id, fold.fold_id)
        probabilities = predict_probabilities(config, dataset, model, boundary, seed)
        inner_evidence.append(
            InnerEvidence(
                fold_id=fold.fold_id,
                estimator_seed=seed,
                probabilities=tuple(
                    ProbabilityObservation(
                        sample_index=index,
                        fold_id=fold.fold_id,
                        truth=int(dataset.target.iloc[index]),
                        probability=p,
                    )
                    for index, p in zip(fold.validation_indices, probabilities, strict=True)
                ),
            )
        )
    observations = sorted(
        (p for f in inner_evidence for p in f.probabilities), key=lambda p: p.sample_index
    )
    selection = select_threshold(
        [p.truth for p in observations], [p.probability for p in observations], config.threshold
    )
    # No outer validation outcomes enter the preceding fit or selection calls.
    seed = estimator_seed(config, identifier, outer_id, None)
    probabilities = predict_probabilities(config, dataset, model, plan.final, seed)
    indices = plan.final.fold.validation_indices
    truth = [int(dataset.target.iloc[i]) for i in indices]
    return OuterEvaluation(
        fold_id=outer_id,
        estimator_seed=seed,
        inner=tuple(inner_evidence),
        selection=selection,
        predictions=tuple(
            DecisionPrediction(
                sample_index=i,
                fold_id=outer_id,
                truth=y,
                probability=p,
                prediction=int(p >= selection.threshold),
                reference_prediction=int(p >= 0.5),
            )
            for i, y, p in zip(indices, truth, probabilities, strict=True)
        ),
        selected=evaluate_policy(truth, probabilities, selection.threshold, config.threshold),
        reference=evaluate_policy(truth, probabilities, 0.5, config.threshold),
        probability_quality=probability_quality(truth, probabilities, config.reliability_bins),
    )


def run_decision_analysis(config: DecisionConfig, *, persist: bool = True) -> DecisionResult:
    """Execute fixed binary candidates; rank only complete honest outer evaluations.

    Providers, custom registries, parallelism, recovery and deployment refits are
    deliberately absent from this independently versioned workflow.
    """
    # Revalidate model_copy inputs and privately snapshot mutable config dictionaries.
    config = DecisionConfig.model_validate(config.model_dump(mode="json"))
    models = _model_identities(config)
    dataset = default_dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError("decision analysis requires a binary-classification dataset")
    if (
        dataset.target_labels is None
        or len(dataset.target_labels) != 2
        or not np.isin(dataset.target.to_numpy(), [0, 1]).all()
    ):
        raise ValueError("binary target requires labels 0/1 and its original label mapping")
    truth = tuple(int(y) for y in dataset.target)
    plan = build_plan(config, truth)
    evidence = DatasetEvidence(
        identity=dataset.identity,
        content_digest=dataset_digest(dataset),
        target_digest=fingerprint(canonical(truth)),
        row_count=dataset.row_count,
        feature_count=dataset.feature_count,
        numeric_features=dataset.numeric_feature_names,
        categorical_features=dataset.categorical_feature_names,
        target_name=dataset.target_name,
        target_labels=dataset.target_labels,
        truth=truth,
    )
    identity = canonical(
        {
            "configuration": config.canonical_dict(include_output=False),
            "dataset": evidence.model_dump(mode="json", exclude={"truth"}),
            "versions": dependency_versions(),
            "models": models,
            "semantics": SEMANTICS,
        }
    )
    candidates = []
    with threadpool_limits(limits=1):
        for model in sorted(config.models, key=lambda m: m.id or m.name):
            completed = []
            failure = None
            for outer in plan.outer:
                try:
                    completed.append(evaluate_outer(config, dataset, model, outer))
                except (ValueError, ArithmeticError) as exc:
                    failure = Failure(
                        fold_id=outer.final.fold.fold_id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                    break
            folds = tuple(completed)
            candidates.append(
                CandidateResult(
                    identifier=model.id or model.name,
                    model_name=model.name,
                    parameters_json=canonical(model.parameters),
                    status="success" if failure is None else ("partial" if completed else "failed"),
                    folds=folds,
                    failure=failure,
                    metrics=aggregate(folds),
                )
            )
    ranked = rank_candidates(tuple(candidates), config.primary_metric)
    statuses = {c.status for c in ranked}
    result = DecisionResult(
        artifact_schema_version=1,
        fingerprint=fingerprint(identity),
        identity_json=identity,
        dataset=evidence,
        plan=plan,
        primary_metric=config.primary_metric,
        optimization_direction=direction(config.primary_metric),
        status="success"
        if statuses == {"success"}
        else ("failed" if statuses == {"failed"} else "partial"),
        candidates=ranked,
    )
    if persist:
        from benchforge.storage.decision import DecisionArtifactStore

        directory = DecisionArtifactStore(config.output.directory).write(config, result)
        result = result.model_copy(update={"artifact_directory": directory})
    if result.status == "failed":
        raise DecisionAnalysisError(result)
    return result
