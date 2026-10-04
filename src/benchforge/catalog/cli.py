"""CLI adapters; indexing and comparison policy live in reusable modules."""

import argparse
import json
from typing import Any

from benchforge.artifacts.types import ArtifactType, VerificationReport
from benchforge.artifacts.verify import verify
from benchforge.catalog.compare import compare
from benchforge.catalog.database import Catalog
from benchforge.catalog.models import CatalogExperiment


def add_commands(subparsers: Any) -> None:
    verification = subparsers.add_parser("verify", help="verify persisted experiment evidence")
    verification.add_argument("directory")
    verification.add_argument(
        "--deep", action="store_true", help="recompute metrics from predictions"
    )
    verification.add_argument("--json", action="store_true")
    catalog = subparsers.add_parser(
        "catalog", help="index and inspect historical experiment evidence"
    )
    catalog.add_argument(
        "--database", default=".benchforge/catalog.sqlite3", help="local SQLite file"
    )
    actions = catalog.add_subparsers(dest="action", required=True)
    for name in ("ingest", "list", "show", "verify", "compare"):
        command = actions.add_parser(name)
        command.add_argument("--json", action="store_true")
        # Permit the database option either before or after the action.
        command.add_argument("--database", default=argparse.SUPPRESS)
        if name == "ingest":
            command.add_argument("directory")
        elif name in {"show", "verify"}:
            command.add_argument("experiment_id")
            if name == "verify":
                command.add_argument("--deep", action="store_true")
        elif name == "compare":
            command.add_argument("experiment_a")
            command.add_argument("experiment_b")
        else:
            command.add_argument("--type", choices=[kind.value for kind in ArtifactType])
            for flag in ("task", "dataset", "metric"):
                command.add_argument(f"--{flag}")
            command.add_argument("--status", choices=["success", "partial", "failed"])


def format_verification(report: VerificationReport) -> str:
    lines = [
        f"verification {'PASSED' if report.passed else 'FAILED'}: {report.directory}",
        f"{report.outcome} integrity={report.integrity}",
    ]
    lines.extend(f"{d.level} [{d.code}]: {d.message}" for d in report.diagnostics)
    return "\n".join(lines)


def _format_experiment(experiment: CatalogExperiment, *, details: bool = False) -> str:
    lines = [
        f"{experiment.experiment_id} {experiment.artifact_type} {experiment.task}",
        f"  dataset={experiment.dataset_identity} "
        f"metric={experiment.primary_metric or 'unavailable'}",
        f"  fingerprint={experiment.fingerprint}",
    ]
    for location in experiment.locations:
        d = location.descriptor
        top = sorted(
            (c for c in d.candidates if c.rank is not None),
            key=lambda c: (c.rank or 0, c.identifier),
        )
        lines.append(
            f"  {location.directory}: status={d.status} failures={d.failure_count} "
            f"verification={location.verification.outcome} "
            f"integrity={location.verification.integrity}"
        )
        if top:
            lines.append(
                f"    top measured candidate={top[0].identifier} "
                f"mean={top[0].metrics.get(d.primary_metric) if d.primary_metric else None}"
            )
        if details:
            lines.append(
                f"    created={d.created_at or 'unavailable'} "
                f"version={d.benchforge_version or 'unavailable'} "
                f"schema={d.artifact_schema_version} seed={d.seed}"
            )
            for candidate in d.candidates:
                lines.append(
                    f"    {candidate.identifier} ({candidate.model_name}): {candidate.status} "
                    f"metrics={json.dumps(candidate.metrics, sort_keys=True)} "
                    f"parameters={json.dumps(candidate.parameters, sort_keys=True)}"
                )
    return "\n".join(lines)


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "verify":
        report = verify(args.directory, deep=args.deep)
        print(report.model_dump_json(indent=2) if args.json else format_verification(report))
        return 0 if report.passed else 2
    with Catalog(args.database) as catalog:
        if args.action == "ingest":
            ids = catalog.ingest(args.directory)
            print(
                json.dumps({"experiment_ids": ids}, indent=2)
                if args.json
                else "Indexed experiments:\n" + "\n".join(ids)
            )
        elif args.action in {"list", "show"}:
            experiments = (
                catalog.list(
                    artifact_type=args.type,
                    task=args.task,
                    dataset=args.dataset,
                    metric=args.metric,
                    status=args.status,
                )
                if args.action == "list"
                else (catalog.show(args.experiment_id),)
            )
            print(
                json.dumps(
                    [item.model_dump(mode="json") for item in experiments], indent=2, sort_keys=True
                )
                if args.json
                else "\n".join(
                    _format_experiment(item, details=args.action == "show") for item in experiments
                )
                or "No experiments."
            )
        elif args.action == "verify":
            reports = catalog.verify(args.experiment_id, deep=args.deep)
            print(
                json.dumps(
                    [report.model_dump(mode="json") for report in reports], indent=2, sort_keys=True
                )
                if args.json
                else "\n".join(format_verification(report) for report in reports)
            )
            return 0 if all(report.passed for report in reports) else 2
        else:
            result = compare(catalog, args.experiment_a, args.experiment_b)
            if args.json:
                print(result.model_dump_json(indent=2))
            else:
                lines = [
                    f"Comparison: {result.compatibility}",
                    f"A: {result.location_a}",
                    f"B: {result.location_b}",
                    *result.reasons,
                ]
                for d in result.differences:
                    lines.append(
                        f"{d.identifier}: A={d.metric_a} B={d.metric_b} "
                        f"B-A={d.raw_delta_b_minus_a} rank={d.rank_a}→{d.rank_b} "
                        f"parameters_changed={d.parameters_changed}"
                    )
                lines.extend(
                    [
                        f"Failures: A={result.failure_count_a} B={result.failure_count_b}",
                        result.interpretation,
                    ]
                )
                print("\n".join(lines))
            return 2 if result.compatibility == "incompatible" else 0
    return 0
