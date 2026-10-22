"""Offline adapter for decision evidence, keeping probabilities out of the HTML."""

from benchforge.artifacts.types import ArtifactDescriptor
from benchforge.artifacts.verify import _decode_json
from benchforge.decision.metrics import direction, fold_metrics
from benchforge.decision.results import DecisionResult
from benchforge.reporting.models import (
    CandidateReport,
    Evidence,
    ExperimentReport,
    FoldReport,
    Metric,
    Score,
    Verification,
)

LIMITATIONS = (
    "Thresholds are selected using inner out-of-fold probabilities only. Inner objective values "
    "are selection evidence, not unbiased evaluation. All reported performance uses untouched "
    "outer validation; choosing a candidate from these results adds selection uncertainty.",
    "The 0.5 reference uses exactly the same outer probabilities as the selected policy. "
    "Lowering a threshold generally trades precision for recall and increases positive decisions. "
    "False-positive and false-negative costs are configured assumptions, not measured utilities.",
    "Discrimination (ROC-AUC) and calibration describe different properties. Brier score and log "
    "loss reflect both probability calibration and discrimination; neither isolates calibration.",
    "ECE is a descriptive, binning-dependent, sample-size-sensitive diagnostic. Empty bins have "
    "unavailable means. Calibration scores do not establish reliability under distribution shift.",
    "Each outer fold trains a different threshold. Their variability is evidence; their average "
    "is not a validated deployment threshold. This workflow produces no deployment model.",
    "Fold standard deviation uses population ddof=0 and is not a confidence interval. Overlapping "
    "training sets do not provide independent replications or statistical significance.",
    "Seals detect byte changes, not authenticity. Offline verification reconstructs scientific "
    "records but cannot prove that training followed the recorded plans or verify source features.",
)


def adapt_decision(d: ArtifactDescriptor, deep: bool) -> ExperimentReport:
    result = DecisionResult.model_validate(_decode_json(d.directory / "decision_result.json"))
    names = sorted({m.name for c in result.candidates for m in c.metrics})
    if not names:
        names = [d.primary_metric] if d.primary_metric else []
    summaries = {
        "calibration": d.configuration["calibration"],
        "threshold": d.configuration["threshold"],
        "reliability_bins": d.configuration["reliability_bins"],
        "candidates": [
            {
                "identifier": c.identifier,
                "metrics": [m.model_dump() for m in c.metrics],
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "selected_threshold": f.selection.threshold,
                        "inner_selection_objective": f.selection.objective_value,
                        "selected": f.selected.model_dump(),
                        "reference": f.reference.model_dump(),
                        "probability_quality": f.probability_quality.model_dump(),
                    }
                    for f in c.folds
                ],
            }
            for c in result.candidates
        ],
    }
    return ExperimentReport(
        directory=d.directory,
        workflow="decision",
        fingerprint=d.fingerprint,
        task=d.task,
        dataset_identity=d.dataset_identity,
        dataset_name=d.dataset_name,
        status=d.status,
        primary_metric=d.primary_metric,
        schema_version=1,
        evaluator_version=d.benchforge_version,
        created_at=d.created_at,
        verification=Verification(
            "sealed" if (d.directory / "manifest.json").exists() else "legacy/unsealed", deep, ()
        ),
        metrics=tuple(Metric(n, n, direction(n), n == d.primary_metric) for n in names),
        candidates=tuple(
            CandidateReport(
                c.identifier,
                c.model_name,
                "fixed-decision",
                c.status,
                c.rank,
                Evidence("configured model parameters", c.parameters_json),
                tuple(Score(m.name, m.mean, m.std) for m in c.metrics),
                "outer folds",
                len(c.folds),
                len(result.plan.outer),
                int(c.failure is not None),
                None,
            )
            for c in sorted(
                result.candidates, key=lambda c: (c.rank is None, c.rank or 0, c.identifier)
            )
        ),
        folds=tuple(
            FoldReport(
                c.identifier,
                f.fold_id,
                None,
                len(result.plan.outer[f.fold_id].final.fold.train_indices),
                len(f.predictions),
                tuple(sorted(fold_metrics(f).items())),
                f.estimator_seed,
                Evidence.capture(
                    "decision_result.json: inner threshold selection", f.selection.model_dump()
                ),
            )
            for c in result.candidates
            for f in c.folds
        ),
        configuration=Evidence.capture("decision_config.json", d.configuration),
        evaluation=Evidence.capture("verified evaluation descriptor", d.evaluation.model_dump()),
        provenance=Evidence.capture("metadata.json", d.source_metadata),
        dataset=Evidence.capture(
            "decision_result.json:dataset (truth omitted)",
            result.dataset.model_dump(exclude={"truth"}),
        ),
        failures=Evidence.capture(
            "decision_result.json:candidates:failure",
            [
                {"model_identifier": c.identifier, **c.failure.model_dump()}
                for c in result.candidates
                if c.failure is not None
            ],
        ),
        final_selection=None,
        plan=None,
        repetition_plan=None,
        repetitions=(),
        pairs=(),
        ranks=(),
        limitations=LIMITATIONS
        + (("Incomplete candidates do not receive ranks.",) if d.status != "success" else ()),
        decision=Evidence.capture(
            "decision_result.json: decision summaries (predictions omitted)", summaries
        ),
    )
