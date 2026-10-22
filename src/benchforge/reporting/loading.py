"""Read-only adapters over the verifier's descriptors and stored JSON summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchforge.artifacts.types import ArtifactDescriptor, ArtifactType
from benchforge.artifacts.verify import _decode_json, verify
from benchforge.catalog.compare import compare_descriptors
from benchforge.core.config import parse_metric
from benchforge.core.provider_config import ProviderManifest
from benchforge.evaluation.metrics import metric_spec
from benchforge.reporting.models import (
    CandidateChange,
    CandidateReport,
    ComparisonReport,
    Evidence,
    ExperimentReport,
    FoldReport,
    Metric,
    ReportingError,
    Score,
    TrialBudget,
    Verification,
    VerificationError,
)
from benchforge.robustness.results import (
    ConfidenceInterval,
    PairwiseComparison,
    RankFrequency,
    RankStability,
)

COMMON_LIMITATIONS = (
    "Results describe the recorded dataset and evaluation procedure. Score differences do not "
    "establish statistical significance, population superiority or causation.",
    "SHA-256 seals detect changes relative to their inventory; they are not cryptographic "
    "signatures of authenticity or proof of source-data provenance.",
    "Missing historical evidence is unavailable. Verification cannot reconstruct evidence "
    "that was never persisted, or guarantee reproducibility across dependency versions.",
)
ROBUSTNESS_LIMITATIONS = (
    "Intervals describe fold/estimator resampling sensitivity on this dataset; "
    "they do not establish population uncertainty or statistical significance.",
    "Repetitions reuse rows and are not independent datasets. Incomplete repetitions are "
    "excluded from paired effects; missing observations are never imputed.",
    "Ranks compare available successful candidates; failures may bias summaries. First-place "
    "and top-two rates use all configured repetitions, including failures. Exact ties share "
    "competition ranks (1, 1, 3); leaderboard positions retain the stored ordering.",
)


def regular_path(value: str | Path) -> Path:
    """Reject symlinks before resolution, including ancestors and dangling links."""
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ReportingError(f"symlink paths are not supported: {path}")
    return path.resolve()


def _verified(directory: str | Path, deep: bool, allow_legacy: bool) -> ArtifactDescriptor:
    path = regular_path(directory)
    result = verify(path, deep=deep)
    if not result.passed or result.descriptor is None:
        details = "; ".join(f"[{d.code}] {d.message}" for d in result.diagnostics)
        raise VerificationError(f"artifact verification FAILED: {path}: {details}")
    if result.integrity != "sealed" and not allow_legacy:
        raise VerificationError(
            f"{path}: legacy/unsealed evidence requires allow_legacy=True or --allow-legacy; "
            "integrity guarantees are unavailable"
        )
    return result.descriptor


def _budget(value: dict[str, Any]) -> TrialBudget | None:
    if "configured_trials" not in value:
        return None
    return TrialBudget(
        *(value[f"{name}_trials"] for name in ("configured", "completed", "failed", "pruned"))
    )


def _scores(value: Any) -> tuple[Score, ...]:
    rows = value if isinstance(value, list) else [dict(name=k, **v) for k, v in value.items()]
    return tuple(
        Score(r["name"], r["mean"], r.get("std")) for r in sorted(rows, key=lambda r: r["name"])
    )


def _fold(
    identifier: str,
    value: dict[str, Any],
    repetition: int | None = None,
    seed: int | None = None,
    selection: Evidence | None = None,
) -> FoldReport:
    metrics = value["metrics"]
    if isinstance(metrics, list):
        metrics = {m["name"]: m["value"] for m in metrics}
    return FoldReport(
        identifier,
        value["fold_id"],
        repetition,
        value["train_size"],
        value["validation_size"],
        tuple(sorted(metrics.items())),
        seed,
        selection,
    )


def _adapt(d: ArtifactDescriptor, deep: bool) -> ExperimentReport:
    path = d.directory
    kind = d.artifact_type
    base = path / "search" if kind == ArtifactType.AUTOML else path
    config = d.configuration

    # Use the verifier's strict decoder; do not call readers that load prediction datasets.
    def read(name: str, *, nested: bool = False) -> Any:
        return _decode_json((base if nested else path) / name)

    science_config = (
        read("search_config.json", nested=True) if kind == ArtifactType.AUTOML else config
    )
    manifests = tuple(
        ProviderManifest.model_validate(m) for m in config.get("provider_manifests", ())
    )
    definitions = tuple(
        Metric(name, spec.display_name, spec.direction, name == d.primary_metric)
        for name in sorted(config["metrics"])
        for spec in [metric_spec(parse_metric(name), manifests)]
    )
    folds: list[FoldReport] = []
    repetitions: list[Evidence] = []
    pairs: list[PairwiseComparison] = []
    ranks: list[RankStability] = []
    plan = repetition_plan = final = None
    if kind == ArtifactType.RUN:
        board = {d.candidates[0].identifier: {"metrics": read("aggregate_metrics.json")}}
        folds.extend(_fold(d.candidates[0].identifier, row) for row in read("fold_metrics.json"))
        failures = []
        dataset = d.source_metadata.get("dataset_summary")
    else:
        board_file = {
            ArtifactType.BENCHMARK: "leaderboard.json",
            ArtifactType.SEARCH: "tuned_leaderboard.json",
            ArtifactType.AUTOML: "tuned_leaderboard.json",
            ArtifactType.ROBUSTNESS: "robustness_leaderboard.json",
        }[kind]
        board = {r["model_identifier"]: r for r in read(board_file, nested=True)}
        failures = read("failures.json", nested=True)
        dataset = read("dataset_summary.json", nested=True)
    if kind in {ArtifactType.SEARCH, ArtifactType.AUTOML}:
        selected = d.summary_evidence.get("final_candidate")
        if selected is not None:
            final = Evidence.capture(
                "result.json:final_candidate (trials omitted)"
                if kind == ArtifactType.AUTOML
                else "final_candidate.json (trials omitted)",
                selected,
            )
        if kind == ArtifactType.AUTOML:
            plan = Evidence.capture("plan.json", read("plan.json"))
        for c in d.candidates:
            if c.status != "success":
                continue
            for number in range(science_config["outer_split"]["n_splits"]):
                prefix = f"families/{c.identifier}/outer_fold_{number}"
                result = read(f"{prefix}/outer_result.json", nested=True)
                selection = Evidence.capture(
                    f"{'search/' if kind == ArtifactType.AUTOML else ''}{prefix}/"
                    "outer_result.json + best_params.json",
                    {
                        **{k: v for k, v in result.items() if k != "fold_result"},
                        "selected_parameters": read(f"{prefix}/best_params.json", nested=True),
                    },
                )
                folds.append(_fold(c.identifier, result["fold_result"], selection=selection))
    elif kind == ArtifactType.BENCHMARK:
        for c in d.candidates:
            if c.status == "success":
                folds.extend(
                    _fold(c.identifier, f) for f in read(f"runs/{c.identifier}/fold_metrics.json")
                )
    elif kind == ArtifactType.ROBUSTNESS:
        rep_plan = read("repetition_plan.json")
        repetition_plan = Evidence.capture("repetition_plan.json", rep_plan)
        for rep in rep_plan["repetitions"]:
            prefix = f"repetitions/repetition_{rep['repetition_id']:03d}"
            repetitions.append(
                Evidence.capture(f"{prefix}/metrics.json", read(f"{prefix}/metrics.json"))
            )
            for c in d.candidates:
                row = read(f"{prefix}/candidates/{c.identifier}/summary.json")
                folds.extend(
                    _fold(c.identifier, f, rep["repetition_id"], row["estimator_seed"])
                    for f in row["fold_results"]
                )
        for p in read("pairwise_comparisons.json"):
            pairs.append(
                PairwiseComparison(
                    **{
                        **p,
                        "interval": ConfidenceInterval(**p["interval"]),
                        **{
                            key: tuple(p[key])
                            for key in ("repetition_ids", "raw_paired_deltas", "paired_deltas")
                        },
                    }
                )
            )
        for r in read("rank_stability.json"):
            ranks.append(
                RankStability(
                    **{
                        **r,
                        "rank_counts": tuple(
                            RankFrequency(int(rank), count)
                            for rank, count in sorted(
                                r["rank_counts"].items(), key=lambda pair: int(pair[0])
                            )
                        ),
                    }
                )
            )
    candidates = []
    for c in sorted(d.candidates, key=lambda c: (c.rank is None, c.rank or 0, c.identifier)):
        row = board[c.identifier]
        robust = kind == ArtifactType.ROBUSTNESS
        expected = (
            science_config["robustness"]["repetitions"]
            if robust
            else science_config.get("outer_split", science_config.get("split", {})).get("n_splits")
        )
        completed = (
            row.get("completed_repetitions")
            if robust
            else (
                sum(f.candidate == c.identifier for f in folds) if c.status == "success" else None
            )
        )
        candidates.append(
            CandidateReport(
                c.identifier,
                c.model_name,
                c.mode,
                c.status,
                c.rank,
                Evidence.capture("configuration candidate parameters", c.parameters),
                _scores(row["metrics"]),
                "repetitions" if robust else "folds",
                completed,
                expected,
                row.get("failed_repetitions")
                if robust
                else sum(f["model_identifier"] == c.identifier for f in failures),
                _budget(row),
            )
        )
    diagnostics: list[tuple[str, str, str]] = []
    # Preserve actual diagnostics rather than implying that a successful seal checks custom code.
    checked = verify(path, deep=deep)
    if not checked.passed or checked.descriptor != d:
        raise VerificationError("artifact changed during report loading; run benchforge verify")
    integrity = checked.integrity
    diagnostics.extend((v.level, v.code, v.message) for v in checked.diagnostics)
    limitations = list(COMMON_LIMITATIONS)
    if kind in {ArtifactType.SEARCH, ArtifactType.AUTOML}:
        limitations.append(
            "Leaderboard scores are untouched outer-fold evaluations of tuning procedures. "
            "Inner-validation and final full-data CV scores select parameters only; they are "
            "not unbiased performance estimates and never rank candidates in this report. "
            "Selecting a family using outer results adds model-selection uncertainty."
        )
    if kind == ArtifactType.ROBUSTNESS:
        limitations.extend(ROBUSTNESS_LIMITATIONS)
    if d.status != "success":
        limitations.append(
            "This experiment is incomplete: failures can change the available candidate pool."
        )
    if d.primary_metric is None:
        limitations.append(
            "No primary metric was declared; no selection criterion or rank is inferred."
        )
    if integrity != "sealed":
        limitations.append(
            "Legacy/unsealed evidence: modern byte-integrity guarantees are unavailable."
        )
    limitations.extend(v[2] for v in diagnostics)
    return ExperimentReport(
        path,
        kind.value,
        d.fingerprint,
        d.task,
        d.dataset_identity,
        d.dataset_name,
        d.status,
        d.primary_metric,
        d.artifact_schema_version,
        d.benchforge_version,
        d.created_at,
        Verification(integrity, deep, tuple(diagnostics)),
        definitions,
        tuple(candidates),
        tuple(folds),
        Evidence.capture("persisted configuration", config),
        Evidence.capture("verified evaluation descriptor", d.evaluation.model_dump(mode="json")),
        Evidence.capture("metadata.json", d.source_metadata),
        Evidence.capture("dataset summary", dataset) if dataset else None,
        Evidence.capture("failures.json", failures),
        final,
        plan,
        repetition_plan,
        tuple(repetitions),
        tuple(pairs),
        tuple(ranks),
        tuple(limitations),
    )


def load_report(
    directory: str | Path, *, deep: bool = False, allow_legacy: bool = False
) -> ExperimentReport:
    """Verify existing evidence and return an immutable presentation model; never train."""
    try:
        return _adapt(_verified(directory, deep, allow_legacy), deep)
    except ReportingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ReportingError(f"cannot load report evidence at {directory}: {exc}") from exc


def load_comparison(
    a: str | Path, b: str | Path, *, deep: bool = False, allow_legacy: bool = False
) -> ComparisonReport:
    """Compare directories using catalog policy without opening a catalog database."""
    da = _verified(a, deep, allow_legacy)
    db = _verified(b, deep, allow_legacy)
    report_a, report_b = _adapt(da, deep), _adapt(db, deep)
    result = compare_descriptors(da, db, "A", "B")
    reasons = list(result.reasons)
    if da.configuration.get("preprocessing") != db.configuration.get("preprocessing"):
        reasons.append("preprocessing settings differ")
    for d in (da, db):
        if not d.evaluation.fold_assignment_digest or not d.evaluation.scikit_learn_version:
            reasons.append("fold assignment or dependency evidence unavailable")
        if not (d.directory / "manifest.json").exists():
            reasons.append("legacy/unsealed integrity guarantees unavailable")
    for key in ("python_version", "optuna_version", "versions"):
        if da.source_metadata.get(key) != db.source_metadata.get(key):
            reasons.append(f"recorded dependency provenance differs: {key}")
    compatibility = result.compatibility
    if reasons and compatibility == "compatible":
        compatibility = "partially compatible"
    changes = tuple(
        CandidateChange(
            c.identifier,
            c.metric_a,
            c.metric_b,
            c.raw_delta_b_minus_a if compatibility == "compatible" else None,
            c.oriented_delta_b_minus_a if compatibility == "compatible" else None,
            c.parameters_changed,
            Evidence.capture("candidate A parameters", c.parameters_a),
            Evidence.capture("candidate B parameters", c.parameters_b),
        )
        for c in result.differences
    )
    return ComparisonReport(
        report_a,
        report_b,
        compatibility,
        tuple(sorted(set(reasons))),
        result.candidates_only_a,
        result.candidates_only_b,
        changes,
        result.interpretation,
    )
