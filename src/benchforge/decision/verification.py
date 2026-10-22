"""Offline schema-1 decision evidence validation; never imports the fitting runner."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from benchforge.artifacts.types import (
    ArtifactDescriptor,
    ArtifactError,
    ArtifactType,
    Candidate,
    EvaluationEvidence,
)
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
    METHODOLOGY_VERSION,
    SEMANTICS,
    build_plan,
    canonical,
    dataset_content_identity,
    estimator_seed,
    fingerprint,
)
from benchforge.decision.results import DecisionResult, ProbabilityObservation


def require(condition: bool, context: str) -> None:
    if not condition:
        raise ArtifactError(f"decision evidence: {context}")


def _numeric_evidence(actual: Any, expected: Any) -> bool:
    """Match counts/structure exactly and recomputed floats at shared verifier tolerance."""
    if isinstance(expected, float):
        return isinstance(actual, float) and math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-12
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_numeric_evidence(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(_numeric_evidence(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def validate_decision(config: DecisionConfig, result: DecisionResult) -> dict[str, Any]:
    """Reconstruct all reproducible scientific evidence, even in standard mode."""
    dataset = result.dataset
    require(len(dataset.truth) == dataset.row_count, "dataset truth/row count disagree")
    require(
        dataset.target_digest == fingerprint(canonical(dataset.truth)),
        "target digest disagrees with stored truth",
    )
    require(
        dataset.feature_count == len(dataset.numeric_features) + len(dataset.categorical_features),
        "feature metadata/count disagree",
    )
    names = dataset.numeric_features + dataset.categorical_features
    require(len(set(names)) == len(names), "duplicate feature metadata")
    require(result.primary_metric == config.primary_metric, "primary metric disagrees")
    require(
        result.optimization_direction == direction(config.primary_metric), "direction disagrees"
    )
    identity: dict[str, Any] = json.loads(result.identity_json)
    require(result.identity_json == canonical(identity), "noncanonical identity snapshot")
    require(result.fingerprint == fingerprint(result.identity_json), "fingerprint disagrees")
    require(
        set(identity) == {"configuration", "dataset", "models", "versions", "semantics"},
        "identity fields disagree",
    )
    require(
        identity["configuration"] == config.canonical_dict(include_output=False),
        "identity configuration disagrees",
    )
    require(
        identity["dataset"] == dataset.model_dump(mode="json", exclude={"truth"}),
        "dataset identity evidence disagrees",
    )
    require(identity["semantics"] == SEMANTICS, "unsupported scientific semantics version")
    versions = identity["versions"]
    require(
        isinstance(versions, dict)
        and set(versions)
        == {
            "benchforge",
            "python",
            "numpy",
            "pandas",
            "scikit_learn",
            "pydantic",
            "scipy",
            "pyarrow",
        }
        and all(isinstance(v, str) and bool(v) for v in versions.values()),
        "missing or invalid dependency identity",
    )
    models = {m.id or m.name: m for m in config.models}
    require(set(identity["models"]) == set(models), "model identity coverage disagrees")
    for identifier, info in identity["models"].items():
        require(
            set(info) == {"class", "resolved_parameters", "registry"}
            and isinstance(info["class"], str)
            and bool(info["class"])
            and isinstance(info["resolved_parameters"], dict)
            and set(info["registry"]) == {models[identifier].name},
            "model implementation identity missing",
        )
    # Reconstruct from stored truth, not original datasets; this also verifies class
    # counts at every boundary, exact seeds, disjointness and training-only nesting.
    require(result.plan == build_plan(config, dataset.truth), "deterministic fold plan disagrees")
    require(
        tuple(c.identifier for c in result.candidates) == tuple(sorted(models)),
        "candidate coverage/order disagrees",
    )
    for candidate in result.candidates:
        identifier = candidate.identifier
        model = models[identifier]
        require(
            candidate.model_name == model.name
            and candidate.parameters_json == canonical(model.parameters),
            f"{identifier}: candidate configuration disagrees",
        )
        completed = len(candidate.folds)
        require(
            [f.fold_id for f in candidate.folds] == list(range(completed)),
            f"{identifier}: completed fold order disagrees",
        )
        if candidate.failure is None:
            require(
                completed == len(result.plan.outer) and candidate.status == "success",
                f"{identifier}: incomplete success",
            )
        else:
            require(
                completed < len(result.plan.outer)
                and candidate.failure.fold_id == completed
                and bool(candidate.failure.error_type)
                and bool(candidate.failure.message)
                and candidate.status == ("partial" if completed else "failed")
                and candidate.rank is None,
                f"{identifier}: failure/completeness disagrees",
            )
        for evaluation, plan in zip(candidate.folds, result.plan.outer, strict=False):
            context = f"{identifier}, outer fold {evaluation.fold_id}"
            require(
                evaluation.estimator_seed
                == estimator_seed(config, identifier, evaluation.fold_id, None),
                f"{context}: estimator seed disagrees",
            )
            require(
                len(evaluation.inner) == len(plan.inner), f"{context}: inner coverage disagrees"
            )
            observations: list[ProbabilityObservation] = []
            for inner, boundary in zip(evaluation.inner, plan.inner, strict=True):
                fold_id = boundary.fold.fold_id
                require(
                    inner.fold_id == fold_id
                    and inner.estimator_seed
                    == estimator_seed(config, identifier, evaluation.fold_id, fold_id),
                    f"{context}: inner fold/seed disagrees",
                )
                _observations(
                    inner.probabilities,
                    boundary.fold.validation_indices,
                    dataset.truth,
                    fold_id,
                    context,
                )
                observations.extend(inner.probabilities)
            observations.sort(key=lambda p: p.sample_index)
            selection = select_threshold(
                [p.truth for p in observations],
                [p.probability for p in observations],
                config.threshold,
            )
            require(evaluation.selection == selection, f"{context}: threshold selection disagrees")
            _observations(
                evaluation.predictions,
                plan.final.fold.validation_indices,
                dataset.truth,
                evaluation.fold_id,
                context,
            )
            for prediction in evaluation.predictions:
                require(
                    prediction.prediction == int(prediction.probability >= selection.threshold)
                    and prediction.reference_prediction == int(prediction.probability >= 0.5),
                    f"{context}: threshold predictions disagree",
                )
            truth = [p.truth for p in evaluation.predictions]
            probabilities = [p.probability for p in evaluation.predictions]
            require(
                evaluation.selected.threshold == selection.threshold
                and _numeric_evidence(
                    evaluation.selected.model_dump(),
                    evaluate_policy(
                        truth, probabilities, selection.threshold, config.threshold
                    ).model_dump(),
                ),
                f"{context}: selected metrics/confusion/cost disagree",
            )
            require(
                evaluation.reference.threshold == 0.5
                and _numeric_evidence(
                    evaluation.reference.model_dump(),
                    evaluate_policy(truth, probabilities, 0.5, config.threshold).model_dump(),
                ),
                f"{context}: reference metrics/confusion/cost disagree",
            )
            require(
                _numeric_evidence(
                    evaluation.probability_quality.model_dump(),
                    probability_quality(truth, probabilities, config.reliability_bins).model_dump(),
                ),
                f"{context}: calibration probability diagnostics disagree",
            )
        require(
            _numeric_evidence(
                [m.model_dump() for m in candidate.metrics],
                [m.model_dump() for m in aggregate(candidate.folds)],
            ),
            f"{identifier}: outer aggregate metrics disagree",
        )
    require(
        result.candidates == rank_candidates(result.candidates, config.primary_metric),
        "complete-only competition ranks disagree",
    )
    statuses = {c.status for c in result.candidates}
    status = (
        "success"
        if statuses == {"success"}
        else ("failed" if statuses == {"failed"} else "partial")
    )
    require(result.status == status, "experiment completeness disagrees")
    return identity


def _observations(
    records: tuple[ProbabilityObservation, ...],
    expected_indices: tuple[int, ...],
    truth: tuple[int, ...],
    fold_id: int,
    context: str,
) -> None:
    require(
        tuple(p.sample_index for p in records) == expected_indices,
        f"{context}: sample assignments/order disagree",
    )
    require(
        all(p.fold_id == fold_id and p.truth == truth[p.sample_index] for p in records),
        f"{context}: truth/fold assignments disagree",
    )


def describe_decision(
    directory: Path,
    config_data: Any,
    result_data: Any,
    metadata: Any,
) -> ArtifactDescriptor:
    """Adapt validated new evidence without reinterpreting any historical fields."""
    require(
        type(config_data.get("schema_version")) is int and config_data["schema_version"] == 1,
        "unsupported configuration schema version",
    )
    require(
        type(result_data.get("artifact_schema_version")) is int
        and result_data["artifact_schema_version"] == 1,
        "unsupported artifact schema version",
    )
    config = DecisionConfig.model_validate(config_data)
    result = DecisionResult.model_validate(result_data)
    require("artifact_directory" not in result_data, "runtime artifact location is not evidence")
    identity = validate_decision(config, result)
    versions = identity["versions"]
    require(
        set(metadata)
        == {
            "artifact_schema_version",
            "benchforge_version",
            "created_at",
            "decision_fingerprint",
            "dataset_identity",
            "methodology_version",
            "versions",
        },
        "metadata fields disagree",
    )
    require(
        type(metadata["artifact_schema_version"]) is int
        and metadata["artifact_schema_version"] == 1,
        "unsupported metadata schema version",
    )
    require(
        metadata["decision_fingerprint"] == result.fingerprint
        and metadata["dataset_identity"] == dataset_content_identity(result.dataset)
        and metadata["versions"] == versions
        and metadata["benchforge_version"] == versions["benchforge"]
        and metadata["methodology_version"] == METHODOLOGY_VERSION,
        "metadata disagrees",
    )
    require(
        datetime.fromisoformat(metadata["created_at"]).tzinfo is not None,
        "creation timestamp requires timezone",
    )
    evaluation = EvaluationEvidence(
        seed=config.seed,
        outer_split=config.outer_split,
        inner_split=config.inner_split,
        methodology_version=METHODOLOGY_VERSION,
        scikit_learn_version=versions["scikit_learn"],
        fold_assignment_digest=fingerprint(canonical(result.plan.model_dump(mode="json"))),
        decision_policy_json=canonical(
            {
                "calibration": config.calibration.model_dump(mode="json"),
                "threshold": config.threshold.model_dump(mode="json"),
                "reliability_bins": config.reliability_bins,
                "preprocessing": config.preprocessing.model_dump(mode="json"),
                "versions": versions,
            }
        ),
    )
    return ArtifactDescriptor(
        artifact_type=ArtifactType.DECISION,
        directory=directory,
        artifact_schema_version=1,
        benchforge_version=versions["benchforge"],
        created_at=metadata["created_at"],
        fingerprint=result.fingerprint,
        dataset_identity=metadata["dataset_identity"],
        dataset_name=config_data["dataset"].get("name"),
        task=config.task,
        seed=config.seed,
        primary_metric=config.primary_metric,
        optimization_direction=result.optimization_direction,
        status=result.status,
        failure_count=sum(c.failure is not None for c in result.candidates),
        candidates=tuple(
            Candidate(
                identifier=c.identifier,
                model_name=c.model_name,
                parameters=json.loads(c.parameters_json),
                mode="fixed-decision",
                status=c.status,
                rank=c.rank,
                metrics={m.name: m.mean for m in c.metrics},
            )
            for c in result.candidates
        ),
        evaluation=evaluation,
        summary_evidence={
            "calibration": config_data["calibration"],
            "threshold": config_data["threshold"],
        },
        configuration=config_data,
        source_metadata=metadata,
    )
