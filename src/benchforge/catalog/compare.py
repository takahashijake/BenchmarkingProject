"""Conservative descriptive comparisons; never treats CV folds as independent samples."""

from typing import Literal

from benchforge.artifacts.types import ArtifactDescriptor
from benchforge.catalog.database import Catalog, CatalogError
from benchforge.catalog.models import CandidateDifference, Comparison


def compare_descriptors(
    a: ArtifactDescriptor, b: ArtifactDescriptor, id_a: str, id_b: str
) -> Comparison:
    incompatible = []
    partial = []
    for label, left_value, right_value in [
        ("workflow", a.artifact_type, b.artifact_type),
        ("task", a.task, b.task),
        ("dataset identity", a.dataset_identity, b.dataset_identity),
        ("primary metric", a.primary_metric, b.primary_metric),
        ("optimization direction", a.optimization_direction, b.optimization_direction),
    ]:
        if left_value != right_value:
            incompatible.append(f"different {label}")
        elif left_value is None:
            partial.append(f"unavailable {label}")
    if a.evaluation != b.evaluation:
        partial.append("evaluation methodology, split seed/settings or dependency versions differ")
    if a.status != "success" or b.status != "success":
        partial.append("failed/partial evidence; missing observations are not imputed")
    models_a = {c.identifier: c for c in a.candidates}
    models_b = {c.identifier: c for c in b.candidates}
    differences = []
    if not incompatible:
        for identifier in sorted(models_a.keys() & models_b.keys()):
            left, right = models_a[identifier], models_b[identifier]
            if left.model_name != right.model_name or left.mode != right.mode:
                partial.append(f"candidate {identifier} has different family/mode")
                continue
            score_a = left.metrics.get(a.primary_metric) if a.primary_metric else None
            score_b = right.metrics.get(b.primary_metric) if b.primary_metric else None
            delta = score_b - score_a if score_a is not None and score_b is not None else None
            if delta is None:
                partial.append(f"candidate {identifier} primary metric unavailable")
            oriented = (
                -delta if a.optimization_direction == "minimize" and delta is not None else delta
            )
            differences.append(
                CandidateDifference(
                    identifier=identifier,
                    metric_a=score_a,
                    metric_b=score_b,
                    raw_delta_b_minus_a=delta,
                    oriented_delta_b_minus_a=oriented,
                    rank_a=left.rank,
                    rank_b=right.rank,
                    parameters_a=left.parameters,
                    parameters_b=right.parameters,
                    parameters_changed=left.parameters != right.parameters,
                )
            )
    if models_a.keys() != models_b.keys():
        partial.append("candidate sets differ; ranks refer to different candidate pools")
    compatibility: Literal["compatible", "partially compatible", "incompatible"] = (
        "incompatible" if incompatible else ("partially compatible" if partial else "compatible")
    )
    return Comparison(
        experiment_a=id_a,
        experiment_b=id_b,
        compatibility=compatibility,
        reasons=tuple(sorted(set([*incompatible, *partial]))),
        fingerprint_changed=a.fingerprint != b.fingerprint,
        candidates_only_a=tuple(sorted(models_a.keys() - models_b.keys())),
        candidates_only_b=tuple(sorted(models_b.keys() - models_a.keys())),
        differences=tuple(differences),
        summary_evidence_a=a.summary_evidence,
        summary_evidence_b=b.summary_evidence,
        failure_count_a=a.failure_count,
        failure_count_b=b.failure_count,
        location_a=a.directory,
        location_b=b.directory,
    )


def compare(catalog: Catalog, id_a: str, id_b: str) -> Comparison:
    for identifier in sorted({id_a, id_b}):
        reports = catalog.verify(identifier)
        if any(not report.passed for report in reports):
            raise CatalogError(f"cannot compare {identifier}: artifact verification FAILED")
    a, b = catalog.show(id_a), catalog.show(id_b)
    result = compare_descriptors(
        a.locations[0].descriptor, b.locations[0].descriptor, a.experiment_id, b.experiment_id
    )
    # Choosing one reproduction must be visible when its measured outcomes differ from other copies.
    reasons = list(result.reasons)
    for experiment in (a, b):
        observations = {
            tuple(c.model_dump_json() for c in location.descriptor.candidates)
            for location in experiment.locations
        }
        if len(observations) > 1:
            reasons.append(
                f"{experiment.experiment_id}: replicas have different measured observations; "
                "using first location in deterministic path order"
            )
    if len(reasons) != len(result.reasons):
        return result.model_copy(
            update={
                "reasons": tuple(sorted(set(reasons))),
                "compatibility": (
                    "incompatible"
                    if result.compatibility == "incompatible"
                    else "partially compatible"
                ),
            }
        )
    return result
