"""Offline evidence verification. No data loading, fitting or source-path dependencies."""

import csv
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ValidationError

from benchforge.artifacts.discovery import CONFIG_FILES, classify, inventory
from benchforge.artifacts.manifest import Manifest, digest
from benchforge.artifacts.types import (
    ArtifactDescriptor,
    ArtifactError,
    ArtifactType,
    Candidate,
    Diagnostic,
    EvaluationEvidence,
    VerificationReport,
)
from benchforge.core.config import (
    AutoMLConfig,
    BenchmarkConfig,
    MetricName,
    RunConfig,
    SearchConfig,
    SplitConfig,
    TaskType,
)
from benchforge.evaluation.metrics import aggregate_fold_metrics, compute_metrics, metric_spec

REQUIRED = {
    ArtifactType.RUN: ("fold_metrics.json", "aggregate_metrics.json", "predictions.csv"),
    ArtifactType.BENCHMARK: (
        "dataset_summary.json",
        "leaderboard.json",
        "leaderboard.csv",
        "failures.json",
    ),
    ArtifactType.SEARCH: (
        "dataset_summary.json",
        "tuned_leaderboard.json",
        "tuned_leaderboard.csv",
        "failures.json",
        "outer_folds.json",
        "final_candidate.json",
    ),
    ArtifactType.AUTOML: ("plan.json", "generated_search_config.json", "result.json"),
    ArtifactType.ROBUSTNESS: (
        "dataset_summary.json",
        "repetition_plan.json",
        "robustness_leaderboard.json",
        "robustness_leaderboard.csv",
        "pairwise_comparisons.json",
        "rank_stability.json",
        "failures.json",
    ),
}
FINGERPRINT_KEYS = {
    ArtifactType.RUN: "config_fingerprint",
    ArtifactType.BENCHMARK: "benchmark_fingerprint",
    ArtifactType.SEARCH: "search_fingerprint",
    ArtifactType.AUTOML: "automl_fingerprint",
    ArtifactType.ROBUSTNESS: "robustness_fingerprint",
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _equal(actual: float, expected: float, context: str) -> None:
    _check(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
        f"{context}: {actual} disagrees with {expected}",
    )


def _identifier(value: Any) -> str:
    _check(
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value,
        f"unsafe candidate identifier: {value!r}",
    )
    return str(value)


def _decode_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ArtifactError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ArtifactError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (ValueError, UnicodeError) as exc:
        raise ArtifactError(f"{path.name}: invalid JSON: {exc}") from exc


class _Evidence:
    """JSON/CSV boundary: untrusted mappings never escape into the catalog domain."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.names = inventory(directory)
        self.json: dict[str, Any] = {}
        for name in self.names:
            path = directory / name
            if path.suffix == ".json":
                self.json[name] = _decode_json(path)
            elif path.suffix == ".csv":
                self.csv(name)

    def get(self, name: str) -> Any:
        if name not in self.json:
            raise ArtifactError(f"missing required JSON file: {name}")
        return self.json[name]

    def require(self, *names: str) -> None:
        for name in names:
            _check(name in self.names, f"missing required file: {name}")

    def csv(self, name: str, *, columns: set[str] | None = None) -> list[dict[str, str]]:
        self.require(name)
        try:
            with (self.directory / name).open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, strict=True)
                _check(
                    bool(reader.fieldnames)
                    and len(set(reader.fieldnames or [])) == len(reader.fieldnames or []),
                    f"{name}: missing/duplicate CSV header",
                )
                _check(
                    columns is None or columns.issubset(reader.fieldnames or []),
                    f"{name}: required CSV columns missing",
                )
                rows = list(reader)
                _check(
                    all(None not in row and None not in row.values() for row in rows),
                    f"{name}: malformed CSV row",
                )
                return rows
        except (csv.Error, UnicodeError) as exc:
            raise ArtifactError(f"{name}: invalid CSV: {exc}") from exc


def _metrics(value: Any) -> dict[str, float]:
    if isinstance(value, list):
        _check(len({row["name"] for row in value}) == len(value), "duplicate metric name")
        return {row["name"]: float(row["mean"]) for row in value}
    _check(isinstance(value, dict), "metrics must be an object or list")
    return {name: float(row["mean"]) for name, row in value.items()}


def _aggregate(folds: list[Any], aggregates: Any, context: str) -> None:
    fold_metrics = [
        {item["name"]: item["value"] for item in fold["metrics"]}
        if isinstance(fold["metrics"], list)
        else fold["metrics"]
        for fold in folds
    ]
    computed = aggregate_fold_metrics(fold_metrics)
    means = _metrics(aggregates)
    _check(set(computed) == set(means), f"{context}: aggregate metric names disagree")
    stds = (
        {row["name"]: row["std"] for row in aggregates}
        if isinstance(aggregates, list)
        else {name: row["std"] for name, row in aggregates.items()}
    )
    for name, metric in computed.items():
        _equal(means[name], metric.mean, f"{context}: {name} mean")
        _equal(float(stds[name]), metric.std, f"{context}: {name} std")


def _folds(folds: list[Any], count: int, context: str) -> None:
    _check(
        sorted(row["fold_id"] for row in folds) == list(range(count)),
        f"{context}: fold identifiers/count disagree",
    )


def _split(folds: list[Any], count: int, context: str, universe: set[int] | None = None) -> None:
    _folds(folds, count, context)
    validation: list[int] = []
    for fold in folds:
        train, valid = fold["train_indices"], fold["validation_indices"]
        _check(
            bool(train)
            and bool(valid)
            and len(set(train)) == len(train)
            and len(set(valid)) == len(valid)
            and not set(train).intersection(valid),
            f"{context}: overlapping, empty or duplicate split indices",
        )
        _check(
            all(isinstance(index, int) and index >= 0 for index in [*train, *valid]),
            f"{context}: invalid split index",
        )
        if universe is not None:
            _check(set(train).union(valid) == universe, f"{context}: split universe disagrees")
        validation.extend(valid)
    _check(len(set(validation)) == len(validation), f"{context}: repeated validation indices")
    if universe is not None:
        _check(set(validation) == universe, f"{context}: incomplete validation coverage")


def _predictions(
    evidence: _Evidence,
    name: str,
    folds: list[Any],
    config: dict[str, Any],
    sample_count: int | None,
    deep: bool,
    complete: bool = True,
    split_plan: list[Any] | None = None,
) -> None:
    required = {"sample_index", "fold_id", "truth", "prediction", "score"}
    rows = evidence.csv(name, columns=required)
    _check(
        all(math.isfinite(float(row[field])) for row in rows for field in ("truth", "prediction"))
        and all(not row["score"] or math.isfinite(float(row["score"])) for row in rows),
        f"{name}: non-finite prediction value",
    )
    _check(not rows or required.issubset(rows[0]), f"{name}: prediction columns missing")
    indices = [int(row["sample_index"]) for row in rows]
    _check(
        len(indices) == len(set(indices))
        and all(i >= 0 and (sample_count is None or i < sample_count) for i in indices),
        f"{name}: duplicate/out-of-range prediction indices",
    )
    if complete and sample_count is not None:
        _check(len(rows) == sample_count, f"{name}: prediction count disagrees with dataset")
    _check(
        set(int(row["fold_id"]) for row in rows).issubset({f["fold_id"] for f in folds}),
        f"{name}: unknown prediction fold",
    )
    for fold in folds:
        subset = [row for row in rows if int(row["fold_id"]) == fold["fold_id"]]
        _check(len(subset) == fold["validation_size"], f"{name}: fold prediction count disagrees")
        if split_plan is not None:
            planned = next(item for item in split_plan if item["fold_id"] == fold["fold_id"])
            _check(
                {int(row["sample_index"]) for row in subset} == set(planned["validation_indices"])
                and fold["train_size"] == len(planned["train_indices"]),
                f"{name}: prediction membership/training size disagrees with split plan",
            )
        if deep:
            task = TaskType(config["task"])
            truth_values = [float(row["truth"]) for row in subset]
            predicted_values = [float(row["prediction"]) for row in subset]
            if task == TaskType.REGRESSION:
                truth = np.asarray(truth_values, dtype=np.float64)
                predicted = np.asarray(predicted_values, dtype=np.float64)
            else:
                _check(
                    all(v.is_integer() for v in [*truth_values, *predicted_values]),
                    f"{name}: classification labels must be integers",
                )
                truth = np.asarray(truth_values, dtype=np.int64)
                predicted = np.asarray(predicted_values, dtype=np.int64)
            scores = (
                np.asarray([float(row["score"]) for row in subset], dtype=np.float64)
                if subset and all(row["score"] for row in subset)
                else None
            )
            recomputed = compute_metrics(
                tuple(MetricName(m) for m in config["metrics"]), truth, predicted, scores, task
            )
            stored = (
                dict((m["name"], m["value"]) for m in fold["metrics"])
                if isinstance(fold["metrics"], list)
                else fold["metrics"]
            )
            for metric, value in recomputed.items():
                _equal(float(stored[metric]), value, f"{name}: fold {fold['fold_id']} {metric}")


def _trial_counts(records: list[Any], summary: dict[str, Any], context: str) -> None:
    for state, key in [
        ("COMPLETE", "completed_trials"),
        ("FAIL", "failed_trials"),
        ("PRUNED", "pruned_trials"),
    ]:
        _check(
            sum(row["state"] == state for row in records) == summary[key],
            f"{context}: {key} disagrees with trial records",
        )
    _check(
        all(row["state"] in {"COMPLETE", "FAIL", "PRUNED"} for row in records),
        f"{context}: unsupported trial state",
    )
    _check(len(records) <= summary["configured_trials"], f"{context}: trial budget exceeded")


def _describe(e: _Evidence, kind: ArtifactType, deep: bool) -> ArtifactDescriptor:
    e.require(CONFIG_FILES[kind], "metadata.json", *REQUIRED[kind])
    config = e.get(CONFIG_FILES[kind])
    metadata = e.get("metadata.json")
    _check(
        isinstance(config, dict) and isinstance(metadata, dict), "config/metadata must be objects"
    )
    _check(
        type(config.get("schema_version")) is int and config["schema_version"] == 1,
        "unsupported configuration schema version",
    )
    _check(
        type(metadata.get("artifact_schema_version", 1)) is int
        and metadata.get("artifact_schema_version", 1) == 1,
        "unsupported artifact schema version",
    )
    if metadata.get("created_at") is not None:
        try:
            created_at = datetime.fromisoformat(metadata["created_at"])
        except (ValueError, TypeError) as exc:
            raise ArtifactError(f"metadata.json: invalid creation timestamp: {exc}") from exc
        _check(created_at.tzinfo is not None, "metadata.json: creation timestamp lacks timezone")
    validators: dict[ArtifactType, type[BaseModel]] = {
        ArtifactType.RUN: RunConfig,
        ArtifactType.BENCHMARK: BenchmarkConfig,
        ArtifactType.SEARCH: SearchConfig,
        ArtifactType.AUTOML: AutoMLConfig,
        ArtifactType.ROBUSTNESS: BenchmarkConfig,
    }
    # Robustness-specific validation is offline here: do not resolve dataset paths or registries.
    validation_config = {key: value for key, value in config.items() if key != "robustness"}
    validators[kind].model_validate(validation_config)
    if kind == ArtifactType.ROBUSTNESS:
        from benchforge.robustness.config import RobustnessSettings

        RobustnessSettings.model_validate(config["robustness"])
    identity = metadata["dataset_identity"]
    fingerprint = metadata[FINGERPRINT_KEYS[kind]]
    _check(isinstance(identity, str) and bool(identity), "missing dataset identity")
    _check(
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(c in "0123456789abcdef" for c in fingerprint),
        "invalid experiment fingerprint",
    )
    dataset: Any = (
        metadata.get("dataset_summary")
        if kind == ArtifactType.RUN
        else (e.get("dataset_summary.json") if kind != ArtifactType.AUTOML else None)
    )
    if dataset:
        _check(
            dataset["identity"] == identity and dataset["task"] == config["task"],
            "dataset summary identity/task disagrees",
        )
    metric = config.get("primary_metric")
    direction = metric_spec(MetricName(metric)).direction if metric else None
    _check(metadata.get("primary_metric", metric) == metric, "primary metric disagrees")
    _check(
        metadata.get("optimization_direction", direction) == direction,
        "optimization direction disagrees with metric semantics",
    )
    _check(metadata.get("seed", config["seed"]) == config["seed"], "seed disagrees")
    candidates: list[Candidate] = []
    failures: list[Any] = []
    evaluation = {
        key: config[key]
        for key in ("split", "outer_split", "inner_split", "robustness")
        if key in config
    }
    evaluation["seed"] = config["seed"]
    evaluation["methodology_version"] = metadata.get("methodology_version")
    fold_assignment: list[Any] = []
    if kind == ArtifactType.RUN:
        fold_assignment = sorted(
            (int(row["sample_index"]), int(row["fold_id"])) for row in e.csv("predictions.csv")
        )
    elif kind == ArtifactType.BENCHMARK:
        success = next(
            (row for row in e.get("leaderboard.json") if row["status"] == "success"), None
        )
        if success:
            fold_assignment = sorted(
                (int(row["sample_index"]), int(row["fold_id"]))
                for row in e.csv(f"runs/{_identifier(success['model_identifier'])}/predictions.csv")
            )
    elif kind == ArtifactType.SEARCH:
        fold_assignment = sorted(
            (index, row["fold_id"])
            for row in e.get("outer_folds.json")
            for index in row["validation_indices"]
        )
    elif kind == ArtifactType.ROBUSTNESS:
        fold_assignment = [
            (
                rep["repetition_id"],
                e.get(f"repetitions/repetition_{rep['repetition_id']:03d}/split_plan.json"),
            )
            for rep in e.get("repetition_plan.json")["repetitions"]
        ]
    if fold_assignment:
        evaluation["fold_assignment_digest"] = hashlib.sha256(
            json.dumps(fold_assignment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    evaluation["scikit_learn_version"] = metadata.get(
        "scikit_learn_version", metadata.get("versions", {}).get("scikit_learn")
    )
    if kind == ArtifactType.RUN:
        model = config["model"]
        folds = e.get("fold_metrics.json")
        _folds(folds, config["split"]["n_splits"], "fold_metrics.json")
        _aggregate(folds, e.get("aggregate_metrics.json"), "aggregate_metrics.json")
        _check(
            dataset is not None or metadata.get("fingerprint_method") is None,
            "metadata.json: resolved run is missing its dataset summary",
        )
        _predictions(
            e, "predictions.csv", folds, config, dataset["row_count"] if dataset else None, deep
        )
        if metadata.get("fingerprint_method") == "resolved-run-v1":
            expected = RunConfig.model_validate(config).fingerprint_for_dataset(identity)
            _check(fingerprint == expected, "resolved run fingerprint disagrees with semantics")
        else:
            _check(metadata.get("fingerprint_method") is None, "unsupported run fingerprint method")
            _check(
                fingerprint == RunConfig.model_validate(config).fingerprint,
                "legacy run fingerprint disagrees with config",
            )
        candidates.append(
            Candidate(
                identifier=_identifier(model.get("id", model["name"])),
                model_name=model["name"],
                parameters=model["parameters"],
                status="success",
                metrics=_metrics(e.get("aggregate_metrics.json")),
            )
        )
    elif kind == ArtifactType.AUTOML:
        nested = verify(e.directory / "search", deep=deep)
        _check(
            nested.passed and nested.descriptor is not None,
            "nested search artifact failed: " + "; ".join(d.message for d in nested.diagnostics),
        )
        child = nested.descriptor
        assert child is not None
        result = e.get("result.json")
        plan = e.get("plan.json")
        if "selected_model_identity" in metadata:
            keys = (
                "total_trial_budget",
                "final_search_trials",
                "outer_search_budget",
                "study_count",
                "trials_per_outer_fold",
                "allocated_outer_trials",
                "allocated_trials",
                "unallocated_trials",
                "approximate_maximum_fits",
            )
            expected = AutoMLConfig.model_validate(config).fingerprint_for_plan(
                dataset_identity=identity,
                selected_models=metadata["selected_model_identity"],
                search_spaces=e.get("search/metadata.json")["search_space_identity"],
                allocation={key: plan[key] for key in keys},
                search_fingerprint=child.fingerprint,
            )
            _check(
                fingerprint == expected, "AutoML fingerprint disagrees with persisted policy/plan"
            )
        _check(
            result["automl_fingerprint"] == fingerprint and plan["fingerprint"] == fingerprint,
            "AutoML fingerprint disagrees across files",
        )
        _check(
            result["search_fingerprint"] == metadata["search_fingerprint"] == child.fingerprint,
            "nested search fingerprint disagrees",
        )
        _check(
            e.get("generated_search_config.json") == e.json["search/search_config.json"],
            "generated search config disagrees with nested search",
        )
        generated = e.get("generated_search_config.json")
        _check(
            sorted(plan["candidates"], key=lambda m: m.get("id", m["name"]))
            == sorted(generated["models"], key=lambda m: m.get("id", m["name"])),
            "AutoML plan candidates disagree with generated search",
        )
        _check(
            plan["dataset_identity"] == identity
            and plan["search_fingerprint"] == child.fingerprint,
            "AutoML plan dataset/search identity disagree",
        )
        studies = len(plan["searchable_families"]) * generated["outer_split"]["n_splits"]
        outer_budget = config["budget"]["total_trials"] - config["budget"]["final_search_trials"]
        _check(
            plan["study_count"] == studies
            and plan["outer_search_budget"] == outer_budget
            and plan["trials_per_outer_fold"] == outer_budget // studies
            and plan["allocated_outer_trials"] == studies * plan["trials_per_outer_fold"]
            and plan["unallocated_trials"] == outer_budget % studies
            and plan["allocated_trials"]
            == plan["allocated_outer_trials"] + plan["final_search_trials"]
            and plan["total_trial_budget"] == config["budget"]["total_trials"]
            and plan["final_search_trials"] == config["budget"]["final_search_trials"],
            "AutoML allocation arithmetic disagrees",
        )
        _check(
            result["leaderboard"] == e.json["search/tuned_leaderboard.json"],
            "AutoML leaderboard disagrees with nested search",
        )
        _check(
            result["final_candidate"] == e.json["search/final_candidate.json"],
            "AutoML final candidate disagrees with nested search",
        )
        _check(
            child.dataset_identity == identity
            and child.task == config["task"]
            and child.seed == config["seed"],
            "AutoML nested semantics disagree",
        )
        metric, direction = child.primary_metric, child.optimization_direction
        _check(
            result["primary_metric"] == metric and result["optimization_direction"] == direction,
            "AutoML result metric/direction disagree",
        )
        candidates = list(child.candidates)
        failures = e.get("search/failures.json")
        evaluation = child.evaluation.model_dump(mode="json")
        dataset = e.get("search/dataset_summary.json")
    else:
        failures = e.get("failures.json")
        board_name = {
            ArtifactType.BENCHMARK: "leaderboard",
            ArtifactType.SEARCH: "tuned_leaderboard",
            ArtifactType.ROBUSTNESS: "robustness_leaderboard",
        }[kind]
        board = e.get(f"{board_name}.json")
        models = {_identifier(m.get("id", m["name"])): m for m in config["models"]}
        ids = [row["model_identifier"] for row in board]
        _check(
            len(ids) == len(set(ids)) and set(ids) == set(models),
            "leaderboard candidates disagree with configured models",
        )
        scored = [row for row in board if row["status"] != "failed"]
        expected_order = sorted(
            scored,
            key=lambda row: (
                {"success": 0, "partial": 1}[row["status"]],
                -float(row["primary_metric_mean"])
                if direction == "maximize"
                else float(row["primary_metric_mean"]),
                row["model_identifier"],
            ),
        )
        _check(
            scored == expected_order and board[: len(scored)] == scored,
            "leaderboard order disagrees with optimization/status semantics",
        )
        _check(
            [row["rank"] for row in scored] == list(range(1, len(scored) + 1)),
            "leaderboard ranks disagree with stored scores",
        )
        csv_board = e.csv(f"{board_name}.csv")
        _check(
            [row["model_identifier"] for row in csv_board] == ids,
            "CSV leaderboard identifiers/order disagree",
        )
        for row in board:
            identifier = _identifier(row["model_identifier"])
            model = models[identifier]
            if kind == ArtifactType.SEARCH:
                _check(row["mode"] == model["mode"], "leaderboard candidate mode disagrees")
            means = _metrics(row["metrics"])
            if row["status"] != "failed":
                _check(set(means) == set(config["metrics"]), "reported metric names disagree")
                _equal(
                    float(row["primary_metric_mean"]), means[metric], "leaderboard primary metric"
                )
            else:
                _check(
                    not means and row["primary_metric_mean"] is None and row["rank"] is None,
                    "failed candidate has aggregate score/rank",
                )
            csv_row = csv_board[ids.index(identifier)]
            _check(csv_row["status"] == row["status"], "CSV leaderboard status disagrees")
            _check(
                csv_row["rank"] == (str(row["rank"]) if row["rank"] is not None else ""),
                "CSV leaderboard rank disagrees",
            )
            if row["primary_metric_std"] is not None:
                _equal(
                    float(csv_row["primary_metric_std"]),
                    float(row["primary_metric_std"]),
                    "CSV leaderboard standard deviation",
                )
            if row["primary_metric_mean"] is not None:
                _equal(
                    float(csv_row["primary_metric_mean"]),
                    float(row["primary_metric_mean"]),
                    "CSV leaderboard metric",
                )
            candidates.append(
                Candidate(
                    identifier=identifier,
                    model_name=model["name"],
                    parameters=model["parameters"],
                    mode=model.get("mode"),
                    status=row["status"],
                    rank=row["rank"],
                    metrics=means,
                )
            )
            if kind == ArtifactType.BENCHMARK and row["status"] == "success":
                child_report = verify(e.directory / "runs" / identifier, deep=deep)
                _check(
                    child_report.passed and child_report.descriptor is not None,
                    f"runs/{identifier}: " + "; ".join(d.message for d in child_report.diagnostics),
                )
                child = child_report.descriptor
                assert child is not None
                _check(
                    child.dataset_identity == identity and child.candidates[0].metrics == means,
                    f"runs/{identifier}: candidate metrics/identity disagree",
                )
                _check(
                    child.candidates[0].parameters == model["parameters"]
                    and child.candidates[0].model_name == model["name"]
                    and child.task == config["task"]
                    and child.seed == config["seed"]
                    and child.evaluation.split == SplitConfig.model_validate(config["split"])
                    and child.evaluation.fold_assignment_digest
                    == evaluation["fold_assignment_digest"],
                    f"runs/{identifier}: configured model/shared folds disagree",
                )
                _check(
                    e.get(f"runs/{identifier}/metadata.json")["benchmark_fingerprint"]
                    == fingerprint,
                    f"runs/{identifier}: parent fingerprint disagrees",
                )
            if kind == ArtifactType.SEARCH:
                _search_candidate(e, identifier, model, row, config, dataset, deep)
        if kind != ArtifactType.ROBUSTNESS:
            failed_ids = [failure["model_identifier"] for failure in failures]
            _check(
                sorted(failed_ids)
                == sorted(c.identifier for c in candidates if c.status == "failed"),
                "failure records disagree with leaderboard",
            )
            prefix = "candidates" if kind == ArtifactType.BENCHMARK else "families"
            _check(
                metadata[f"failed_{prefix}"] == len(failures)
                and metadata[f"successful_{prefix}"] == len(board) - len(failures),
                "metadata failure/success counts disagree",
            )
        if kind == ArtifactType.BENCHMARK:
            if metadata.get("fingerprint_method") == "canonical-metrics-v1":
                expected = BenchmarkConfig.model_validate(config).fingerprint_for_dataset(identity)
            else:
                _check(metadata.get("fingerprint_method") is None, "unsupported fingerprint method")
                semantics = BenchmarkConfig.model_validate(config).canonical_dict(
                    include_output=False
                )
                semantics["dataset"].pop("path", None)
                payload = {"benchmark": semantics, "resolved_dataset_identity": identity}
                expected = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            _check(fingerprint == expected, "benchmark fingerprint disagrees with semantics")
        if kind == ArtifactType.SEARCH:
            if "search_space_identity" in metadata:
                expected = SearchConfig.model_validate(config).fingerprint_for_dataset(
                    identity, metadata["search_space_identity"]
                )
                _check(
                    fingerprint == expected, "search fingerprint disagrees with persisted semantics"
                )
            outer = e.get("outer_folds.json")
            _split(
                outer,
                config["outer_split"]["n_splits"],
                "outer_folds.json",
                set(range(dataset["row_count"])),
            )
            final = e.get("final_candidate.json")
            if final is not None:
                _check(
                    final["model_identifier"] in models, "final candidate references unknown family"
                )
                _trial_counts(final["trials"], final, "final_candidate.json")
                winner = next(
                    row["model_identifier"] for row in board if row["status"] == "success"
                )
                _check(
                    final["model_identifier"] == winner
                    and final["model_name"] == models[winner]["name"]
                    and final["mode"] == models[winner]["mode"],
                    "final candidate does not reference the top measured family",
                )
        if kind == ArtifactType.ROBUSTNESS:
            if "model_identity" in metadata:
                semantics = dict(config)
                semantics.pop("output", None)
                semantics["dataset"] = {
                    k: v for k, v in semantics["dataset"].items() if k != "path"
                }
                payload = {
                    "robustness": semantics,
                    "dataset_identity": identity,
                    "models": metadata["model_identity"],
                    "versions": metadata["versions"],
                    "methodology_version": metadata["methodology_version"],
                }
                expected = hashlib.sha256(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest()
                _check(
                    fingerprint == expected,
                    "robustness fingerprint disagrees with persisted semantics",
                )
            _robustness(e, config, board, failures, dataset, deep)
    statuses = {candidate.status for candidate in candidates}
    status: Literal["success", "partial", "failed"] = (
        "failed"
        if statuses == {"failed"}
        else ("partial" if failures or statuses != {"success"} else "success")
    )
    return ArtifactDescriptor(
        artifact_type=kind,
        directory=e.directory,
        artifact_schema_version=metadata.get("artifact_schema_version"),
        benchforge_version=metadata.get(
            "benchforge_version", metadata.get("versions", {}).get("benchforge")
        ),
        created_at=metadata.get("created_at"),
        fingerprint=fingerprint,
        dataset_identity=identity,
        dataset_name=config["dataset"].get("name"),
        task=config["task"],
        seed=config["seed"],
        primary_metric=metric,
        optimization_direction=direction,
        status=status,
        failure_count=len(failures),
        candidates=tuple(sorted(candidates, key=lambda c: c.identifier)),
        evaluation=EvaluationEvidence.model_validate(evaluation),
        summary_evidence=_summary_evidence(e, kind),
        configuration=config,
        source_metadata=metadata,
    )


def _summary_evidence(e: _Evidence, kind: ArtifactType) -> dict[str, Any]:
    if kind == ArtifactType.ROBUSTNESS:
        return {
            "pairwise_comparisons": [
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"raw_paired_deltas", "paired_deltas", "repetition_ids"}
                }
                for row in e.get("pairwise_comparisons.json")
            ],
            "rank_stability": e.get("rank_stability.json"),
            "leaderboard": e.get("robustness_leaderboard.json"),
        }
    if kind in {ArtifactType.SEARCH, ArtifactType.AUTOML}:
        final = (
            e.get("final_candidate.json")
            if kind == ArtifactType.SEARCH
            else e.get("result.json")["final_candidate"]
        )
        return {
            "final_candidate": {k: v for k, v in final.items() if k != "trials"} if final else None
        }
    return {}


def _search_candidate(
    e: _Evidence,
    identifier: str,
    model: dict[str, Any],
    row: dict[str, Any],
    config: dict[str, Any],
    dataset: dict[str, Any],
    deep: bool,
) -> None:
    base = f"families/{identifier}"
    if row["status"] == "failed":
        failure = e.get(f"{base}/failure.json")
        _check(failure in e.get("failures.json"), f"{base}: failure summary disagrees")
        _trial_counts(e.get(f"{base}/trials.json"), failure, base)
        e.require(f"{base}/trials.csv")
        return
    summary = e.get(f"{base}/summary.json")
    _check(
        summary["model_identifier"] == identifier and summary["candidate"] == model,
        f"{base}: configured candidate disagrees with summary",
    )
    _check(summary["aggregate_metrics"] == row["metrics"], f"{base}: aggregate metrics disagree")
    folds = []
    counts = {
        key: 0
        for key in ("configured_trials", "completed_trials", "failed_trials", "pruned_trials")
    }
    for outer in e.get("outer_folds.json"):
        path = f"{base}/outer_fold_{outer['fold_id']}"
        result = e.get(f"{path}/outer_result.json")
        split = e.get(f"{path}/split_plan.json")
        _check(
            split["outer_train_indices"] == outer["train_indices"]
            and split["outer_validation_indices"] == outer["validation_indices"],
            f"{path}: outer split disagrees",
        )
        if model["mode"] == "search":
            _split(
                split["inner_folds"],
                config["inner_split"]["n_splits"],
                path,
                set(outer["train_indices"]),
            )
        else:
            _check(not split["inner_folds"], f"{path}: fixed family has inner folds")
        trials = e.get(f"{path}/trials.json")
        _trial_counts(trials, result, path)
        expected_budget = (
            config["budget"]["trials_per_outer_fold"] if model["mode"] == "search" else 0
        )
        _check(
            result["configured_trials"] == expected_budget,
            f"{path}: configured trial budget disagrees",
        )
        _check(
            [trial["number"] for trial in trials] == list(range(len(trials))),
            f"{path}: trial identifiers disagree",
        )
        best_parameters = e.get(f"{path}/best_params.json")
        _check(
            all(best_parameters.get(k) == v for k, v in model["parameters"].items()),
            f"{path}: search overwrote fixed parameters",
        )
        if model["mode"] == "search":
            completed = [trial for trial in trials if trial["state"] == "COMPLETE"]
            _check(bool(completed), f"{path}: successful search has no completed trials")
            select = (
                max
                if metric_spec(MetricName(config["primary_metric"])).direction == "maximize"
                else min
            )
            best_score = select(float(trial["value"]) for trial in completed)
            _equal(float(result["best_inner_score"]), best_score, f"{path}: inner selection score")
            _check(
                any(
                    trial["parameters"] == best_parameters and trial["value"] == best_score
                    for trial in completed
                ),
                f"{path}: selected parameters disagree with best trial",
            )
        else:
            _check(best_parameters == model["parameters"], f"{path}: fixed parameters disagree")
        e.require(f"{path}/trials.csv", f"{path}/best_params.json")
        _check(
            len(e.csv(f"{path}/trials.csv")) == len(trials), f"{path}: trial CSV count disagrees"
        )
        fold = result["fold_result"]
        _check(fold["fold_id"] == outer["fold_id"], f"{path}: fold identifier disagrees")
        folds.append(fold)
        for key in counts:
            counts[key] += result[key]
    for key, value in counts.items():
        _check(summary[key] == row[key] == value, f"{base}: {key} disagrees")
    _aggregate(folds, summary["aggregate_metrics"], base)
    _predictions(
        e,
        f"{base}/outer_predictions.csv",
        folds,
        config,
        dataset["row_count"],
        deep,
        split_plan=e.get("outer_folds.json"),
    )


def _robustness(
    e: _Evidence,
    config: dict[str, Any],
    board: list[Any],
    failures: list[Any],
    dataset: dict[str, Any],
    deep: bool,
) -> None:
    from benchforge.robustness.config import RobustnessSettings
    from benchforge.robustness.results import CandidateScore, RepetitionScores
    from benchforge.robustness.statistics import compare_all_pairs, rank_repetition, summarize_ranks

    direction = metric_spec(MetricName(config["primary_metric"])).direction
    observations: list[RepetitionScores] = []
    plan = e.get("repetition_plan.json")
    ids = [row["repetition_id"] for row in plan["repetitions"]]
    count = config["robustness"]["repetitions"]
    _check(ids == list(range(count)), "repetition plan count/identifiers disagree")
    models = {_identifier(m.get("id", m["name"])): m for m in config["models"]}
    _check(
        sorted(plan["candidate_identifiers"]) == sorted(models), "repetition candidates disagree"
    )
    scores: dict[str, list[float]] = {identifier: [] for identifier in models}
    recorded_failures = []
    for rep in ids:
        base = f"repetitions/repetition_{rep:03d}"
        split = e.get(f"{base}/split_plan.json")
        _split(split, config["split"]["n_splits"], base, set(range(dataset["row_count"])))
        e.require(f"{base}/ranks.json", f"{base}/metrics.json")
        rep_metrics = e.get(f"{base}/metrics.json")
        _check(
            sorted(m["model_identifier"] for m in rep_metrics) == sorted(models),
            f"{base}: repetition metric candidates disagree",
        )
        repetition_scores: list[CandidateScore] = []
        for identifier in models:
            path = f"{base}/candidates/{identifier}"
            summary = e.get(f"{path}/summary.json")
            _check(
                summary["model_identifier"] == identifier
                and summary["model_name"] == models[identifier]["name"],
                f"{path}: candidate summary identifier/name disagree",
            )
            folded = summary["fold_results"]
            _check(len({f["fold_id"] for f in folded}) == len(folded), f"{path}: duplicate folds")
            entry = next(m for m in rep_metrics if m["model_identifier"] == identifier)
            _check(
                entry["metrics"] == summary["metrics"] and entry["status"] == summary["status"],
                f"{path}: repetition summary metrics/status disagree",
            )
            if summary["status"] == "success":
                _folds(folded, len(split), path)
                _aggregate(folded, summary["metrics"], path)
                score = _metrics(summary["metrics"])[config["primary_metric"]]
                scores[identifier].append(score)
                repetition_scores.append(CandidateScore(identifier, score))
                _check(summary["failure"] is None, f"{path}: successful candidate has failure")
            else:
                _check(
                    summary["status"] == "failed" and not summary["metrics"],
                    f"{path}: invalid failed candidate",
                )
                failure = summary["failure"]
                _check(
                    failure["repetition_id"] == rep and failure["model_identifier"] == identifier,
                    f"{path}: failure identity disagrees",
                )
                _check(
                    [f["fold_id"] for f in folded] == list(range(failure["fold_id"]))
                    and failure["skipped_fold_ids"]
                    == list(range(failure["fold_id"] + 1, len(split))),
                    f"{path}: failed/skipped fold evidence disagrees",
                )
                recorded_failures.append(failure)
            _predictions(
                e,
                f"{path}/predictions.csv",
                folded,
                config,
                dataset["row_count"],
                deep,
                complete=summary["status"] == "success",
                split_plan=split,
            )
        observation = RepetitionScores(rep, tuple(repetition_scores))
        observations.append(observation)
        _check(
            e.get(f"{base}/ranks.json")
            == [asdict(r) for r in rank_repetition(observation, direction)],
            f"{base}: stored ranks disagree with repetition scores",
        )
    reconstructed_ranks = [
        {**asdict(row), "rank_counts": {str(r.rank): r.count for r in row.rank_counts}}
        for row in summarize_ranks(tuple(sorted(models)), tuple(observations), direction)
    ]
    _check(
        e.get("rank_stability.json") == reconstructed_ranks,
        "rank stability disagrees with repetition evidence",
    )
    _check(
        sorted(failures, key=lambda f: (f["repetition_id"], f["model_identifier"]))
        == sorted(recorded_failures, key=lambda f: (f["repetition_id"], f["model_identifier"])),
        "robustness failures disagree with candidate summaries",
    )
    for row in board:
        values = scores[row["model_identifier"]]
        _check(
            row["completed_repetitions"] == len(values)
            and row["failed_repetitions"] == count - len(values),
            "leaderboard repetition counts disagree",
        )
        expected_status = "success" if len(values) == count else ("partial" if values else "failed")
        _check(row["status"] == expected_status, "robustness leaderboard status disagrees")
        if values:
            _equal(float(row["primary_metric_mean"]), float(np.mean(values)), "robustness mean")
            _equal(
                float(row["primary_metric_std"]),
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "robustness std",
            )
    pairs = e.get("pairwise_comparisons.json")
    expected_pairs = {(a, b) for a in models for b in models if a < b}
    _check(
        {(p["candidate_a"], p["candidate_b"]) for p in pairs} == expected_pairs
        and len(pairs) == len(expected_pairs),
        "pairwise candidate coverage disagrees",
    )
    for pair in pairs:
        n = pair["usable_repetitions"]
        _check(
            n
            == len(pair["repetition_ids"])
            == len(pair["raw_paired_deltas"])
            == len(pair["paired_deltas"])
            and pair["a_wins"] + pair["b_wins"] + pair["ties"] == n,
            "pairwise observation counts disagree",
        )

        observed_ids = []
        raw = []
        for observation in observations:
            scores_by_id = {item.model_identifier: item.value for item in observation.scores}
            if pair["candidate_a"] in scores_by_id and pair["candidate_b"] in scores_by_id:
                observed_ids.append(observation.repetition_id)
                raw.append(scores_by_id[pair["candidate_a"]] - scores_by_id[pair["candidate_b"]])
        oriented = [value if direction == "maximize" else -value for value in raw]
        _check(
            pair["primary_metric"] == config["primary_metric"]
            and pair["optimization_direction"] == direction
            and pair["repetition_ids"] == observed_ids
            and pair["raw_paired_deltas"] == raw
            and pair["paired_deltas"] == oriented,
            "paired deltas disagree with repetition evidence",
        )
    if deep:
        expected = [
            json.loads(json.dumps(asdict(pair)))
            for pair in compare_all_pairs(
                tuple(sorted(models)),
                tuple(observations),
                config["primary_metric"],
                direction,
                RobustnessSettings.model_validate(config["robustness"]),
                config["seed"],
            )
        ]
        _check(pairs == expected, "pairwise/bootstrap summaries disagree with repetition evidence")


def verify(
    directory: str | Path, *, deep: bool = False, _allow_unsealed: bool = False
) -> VerificationReport:
    path = Path(directory).absolute()
    if not path.is_symlink():
        path = path.resolve()
    diagnostics: list[Diagnostic] = []
    descriptor = None
    manifest = None
    integrity: Literal["sealed", "legacy/unsealed", "invalid"] = "invalid"
    errors = (
        ArithmeticError,
        ArtifactError,
        ValidationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        StopIteration,
        AttributeError,
    )
    # Check bytes even when semantic parsing fails, so corruption names the changed member.
    try:
        _check(not path.is_symlink() and path.is_dir(), f"unsupported artifact directory: {path}")
        names = inventory(path)
        if (path / "manifest.json").exists():
            integrity = "invalid"
            manifest_data = _decode_json(path / "manifest.json")
            _check(
                isinstance(manifest_data, dict)
                and manifest_data.get("manifest_schema_version") == 1,
                "unsupported manifest schema version",
            )
            manifest = Manifest.model_validate(manifest_data)
            declared = {item.path for item in manifest.files}
            actual = set(names)
            for name in sorted(declared - actual):
                diagnostics.append(
                    Diagnostic(
                        level="FAIL",
                        code="missing_member",
                        file=name,
                        message=f"manifest member missing: {name}",
                    )
                )
            for name in sorted(actual - declared):
                diagnostics.append(
                    Diagnostic(
                        level="FAIL",
                        code="unexpected_member",
                        file=name,
                        message=f"unexpected unprotected file: {name}",
                    )
                )
            for item in manifest.files:
                if item.path in actual and digest(path / item.path) != item.sha256:
                    diagnostics.append(
                        Diagnostic(
                            level="FAIL",
                            code="hash_mismatch",
                            file=item.path,
                            message=f"SHA-256 mismatch: {item.path}",
                        )
                    )
    except errors as exc:
        diagnostics.append(
            Diagnostic(level="FAIL", code="invalid_manifest_or_structure", message=f"{path}: {exc}")
        )
    try:
        kind = classify(path)
        evidence = _Evidence(path)
        descriptor = _describe(evidence, kind, deep)
        if manifest is None and descriptor.benchforge_version is not None and not _allow_unsealed:
            version_parts = descriptor.benchforge_version.split(".")
            if len(version_parts) >= 2 and all(part.isdigit() for part in version_parts[:2]):
                _check(
                    tuple(int(part) for part in version_parts[:2]) < (0, 8),
                    "V0.8 artifact is incomplete: required manifest.json is missing",
                )
        if kind == ArtifactType.RUN and "dataset_summary" not in descriptor.source_metadata:
            diagnostics.append(
                Diagnostic(
                    level="WARNING",
                    code="historical_metadata_unavailable",
                    message="dataset summary unavailable; dataset row bounds cannot be verified",
                )
            )
        if manifest is None and not any(d.level == "FAIL" for d in diagnostics):
            integrity = "legacy/unsealed"
            diagnostics.append(
                Diagnostic(
                    level="WARNING",
                    code="legacy_unsealed",
                    message="legacy/unsealed: cryptographic integrity unavailable",
                )
            )
        if manifest is not None:
            _check(manifest.artifact_type == kind, "manifest artifact type disagrees")
            _check(
                manifest.experiment_fingerprint == descriptor.fingerprint,
                "manifest fingerprint disagrees with artifact metadata",
            )
            _check(
                manifest.created_at == descriptor.created_at
                and manifest.benchforge_version == descriptor.benchforge_version,
                "manifest creation/version metadata disagrees",
            )
        if not any(item.level == "FAIL" for item in diagnostics) and manifest is not None:
            integrity = "sealed"
    except errors as exc:
        message = str(exc)
        code = (
            "unsupported_schema"
            if "unsupported" in message and "version" in message
            else (
                "unsupported_artifact" if "unsupported artifact" in message else "invalid_evidence"
            )
        )
        diagnostics.append(Diagnostic(level="FAIL", code=code, message=f"{path}: {message}"))
    return VerificationReport(
        directory=path,
        descriptor=descriptor,
        integrity=integrity,
        diagnostics=tuple(diagnostics),
        deep=deep,
    )
