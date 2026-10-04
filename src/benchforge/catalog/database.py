"""Small versioned SQLite index with atomic, idempotent ingestion."""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from benchforge.artifacts.discovery import discover, inventory
from benchforge.artifacts.manifest import digest
from benchforge.artifacts.types import ArtifactDescriptor, ArtifactError, VerificationReport
from benchforge.artifacts.verify import verify
from benchforge.catalog.models import CatalogExperiment, CatalogLocation

SCHEMA_VERSION = 1
DDL = (
    "CREATE TABLE experiments (id TEXT PRIMARY KEY, artifact_type TEXT NOT NULL, "
    "fingerprint TEXT NOT NULL, task TEXT NOT NULL, dataset_identity TEXT NOT NULL, "
    "primary_metric TEXT, semantics TEXT NOT NULL, UNIQUE(artifact_type, fingerprint))",
    "CREATE TABLE artifact_locations (id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL "
    "REFERENCES experiments(id), directory TEXT NOT NULL UNIQUE, descriptor TEXT NOT NULL, "
    "evidence_digest TEXT NOT NULL)",
    "CREATE TABLE candidates (location_id INTEGER NOT NULL REFERENCES artifact_locations(id) "
    "ON DELETE CASCADE, identifier TEXT NOT NULL, model_name TEXT NOT NULL, status TEXT NOT NULL, "
    "rank INTEGER, parameters TEXT NOT NULL, PRIMARY KEY(location_id, identifier))",
    "CREATE TABLE metrics (location_id INTEGER NOT NULL, candidate_identifier TEXT NOT NULL, "
    "name TEXT NOT NULL, value REAL NOT NULL, "
    "PRIMARY KEY(location_id, candidate_identifier, name), "
    "FOREIGN KEY(location_id, candidate_identifier) REFERENCES candidates(location_id, identifier) "
    "ON DELETE CASCADE)",
    "CREATE TABLE verification_records (id INTEGER PRIMARY KEY, location_id INTEGER NOT NULL "
    "REFERENCES artifact_locations(id) ON DELETE CASCADE, checked_at TEXT NOT NULL, "
    "report TEXT NOT NULL)",
    "CREATE INDEX verification_by_location ON verification_records(location_id, id)",
)


def experiment_id(descriptor: ArtifactDescriptor) -> str:
    return hashlib.sha256(
        f"{descriptor.artifact_type.value}:{descriptor.fingerprint}".encode()
    ).hexdigest()


def _semantics(descriptor: ArtifactDescriptor) -> str:
    # Scientific results, locations, runtime/version annotations and timestamps are observations.
    config = json.loads(json.dumps(descriptor.configuration))
    config.pop("output", None)
    config["dataset"].pop("path", None)
    config["metrics"] = sorted(config["metrics"])
    if isinstance(config.get("models"), list):
        config["models"] = sorted(config["models"], key=lambda m: m.get("id", m["name"]))
    value = {
        "configuration": config,
        "task": descriptor.task,
        "dataset_identity": descriptor.dataset_identity,
        "primary_metric": descriptor.primary_metric,
        "optimization_direction": descriptor.optimization_direction,
        "seed": descriptor.seed,
        "evaluation": {
            k: v
            for k, v in descriptor.evaluation.model_dump(mode="json").items()
            if k not in {"scikit_learn_version", "fold_assignment_digest"}
        },
        "candidates": [
            {
                "identifier": c.identifier,
                "model_name": c.model_name,
                "parameters": c.parameters,
                "mode": c.mode,
            }
            for c in descriptor.candidates
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _evidence_digest(path: Path) -> str:
    files = [(name, digest(path / name)) for name in inventory(path)]
    if (path / "manifest.json").exists():
        files.append(("manifest.json", digest(path / "manifest.json")))
    return hashlib.sha256(json.dumps(sorted(files), separators=(",", ":")).encode()).hexdigest()


class CatalogError(RuntimeError):
    """Database/schema/identity error, distinct from invalid artifact evidence."""


def _database_operation[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except sqlite3.Error as exc:
            raise CatalogError(f"catalog {function.__name__} failed: {exc}") from exc

    return wrapped


class Catalog:
    def __init__(self, path: str | Path = ".benchforge/catalog.sqlite3") -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            with self.connection:
                # Lock initialization before checking the version, including concurrent creators.
                self.connection.execute("BEGIN IMMEDIATE")
                version = self.connection.execute("PRAGMA user_version").fetchone()[0]
                tables = self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if version == 0 and not tables:
                    for statement in DDL:
                        self.connection.execute(statement)
                    self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise CatalogError(f"unsupported catalog schema version: {version}")
        except (sqlite3.Error, OSError, CatalogError) as exc:
            if hasattr(self, "connection"):
                self.connection.close()
            raise CatalogError(f"cannot open catalog {self.path}: {exc}") from exc

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *args: object) -> None:
        self.connection.close()

    @_database_operation
    def ingest(self, root: str | Path) -> tuple[str, ...]:
        discovery = discover(root, validate=False)
        errors = [item.message for item in discovery.diagnostics if item.level == "FAIL"]
        if errors:
            raise ArtifactError("; ".join(errors))
        if not discovery.directories:
            raise ArtifactError(f"no BenchForge artifacts found under {root}")
        # Validate the complete batch before modifying the index. Failed batches roll back fully.
        prepared = []
        for directory in discovery.directories:
            before = _evidence_digest(directory)
            report = verify(directory, deep=True)
            if not report.passed or report.descriptor is None:
                raise ArtifactError(
                    f"verification FAILED for {directory}: "
                    + "; ".join(d.message for d in report.diagnostics if d.level == "FAIL")
                )
            if _evidence_digest(directory) != before:
                raise ArtifactError(f"artifact changed during ingestion: {directory}")
            prepared.append((report, before))
        ids = []
        try:
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                for report, evidence_digest in prepared:
                    ids.append(self._ingest_verified(report, evidence_digest))
        except sqlite3.Error as exc:
            raise CatalogError(f"catalog ingestion transaction failed: {exc}") from exc
        return tuple(sorted(set(ids)))

    def _ingest_verified(self, report: VerificationReport, evidence_digest: str) -> str:
        descriptor = report.descriptor
        assert descriptor is not None
        identifier = experiment_id(descriptor)
        semantics = _semantics(descriptor)
        previous = self.connection.execute(
            "SELECT semantics FROM experiments WHERE id=?", (identifier,)
        ).fetchone()
        if previous is not None and previous[0] != semantics:
            raise CatalogError(
                f"semantic fingerprint collision for {identifier}; "
                "stored identity disagrees with artifact semantics"
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identifier,
                descriptor.artifact_type.value,
                descriptor.fingerprint,
                descriptor.task,
                descriptor.dataset_identity,
                descriptor.primary_metric,
                semantics,
            ),
        )
        location = self.connection.execute(
            "SELECT id, experiment_id, evidence_digest FROM artifact_locations WHERE directory=?",
            (str(descriptor.directory.resolve()),),
        ).fetchone()
        if location is not None:
            if location[1] != identifier or location[2] != evidence_digest:
                raise CatalogError(
                    f"indexed artifact changed at {descriptor.directory}; "
                    "preserve immutable evidence and ingest a new location"
                )
            return identifier
        cursor = self.connection.execute(
            "INSERT INTO artifact_locations "
            "(experiment_id, directory, descriptor, evidence_digest) "
            "VALUES (?, ?, ?, ?)",
            (
                identifier,
                str(descriptor.directory.resolve()),
                descriptor.model_dump_json(),
                evidence_digest,
            ),
        )
        location_id = cursor.lastrowid
        for candidate in descriptor.candidates:
            self.connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                (
                    location_id,
                    candidate.identifier,
                    candidate.model_name,
                    candidate.status,
                    candidate.rank,
                    json.dumps(candidate.parameters, sort_keys=True),
                ),
            )
            for name, value in sorted(candidate.metrics.items()):
                self.connection.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                    (location_id, candidate.identifier, name, value),
                )
        self._record_verification(location_id, report)
        return identifier

    def _record_verification(self, location_id: int | None, report: VerificationReport) -> None:
        self.connection.execute(
            "INSERT INTO verification_records (location_id, checked_at, report) VALUES (?, ?, ?)",
            (location_id, datetime.now(UTC).isoformat(), report.model_dump_json()),
        )

    def _resolve(self, identifier: str) -> str:
        rows = self.connection.execute("SELECT id FROM experiments ORDER BY id").fetchall()
        matches = [str(row[0]) for row in rows if str(row[0]).startswith(identifier)]
        if not identifier or len(matches) != 1:
            raise CatalogError(f"unknown or ambiguous experiment ID: {identifier}")
        return matches[0]

    @_database_operation
    def show(self, identifier: str) -> CatalogExperiment:
        resolved = self._resolve(identifier)
        row = self.connection.execute(
            "SELECT * FROM experiments WHERE id=?", (resolved,)
        ).fetchone()
        locations = []
        for item in self.connection.execute(
            "SELECT * FROM artifact_locations WHERE experiment_id=? ORDER BY directory", (resolved,)
        ):
            report_row = self.connection.execute(
                "SELECT report FROM verification_records WHERE location_id=? "
                "ORDER BY id DESC LIMIT 1",
                (item["id"],),
            ).fetchone()
            locations.append(
                CatalogLocation(
                    directory=Path(item["directory"]),
                    descriptor=ArtifactDescriptor.model_validate_json(item["descriptor"]),
                    verification=VerificationReport.model_validate_json(report_row[0]),
                    evidence_digest=item["evidence_digest"],
                )
            )
        return CatalogExperiment(
            experiment_id=resolved,
            artifact_type=row["artifact_type"],
            fingerprint=row["fingerprint"],
            task=row["task"],
            dataset_identity=row["dataset_identity"],
            primary_metric=row["primary_metric"],
            locations=tuple(locations),
        )

    @_database_operation
    def list(
        self,
        *,
        artifact_type: str | None = None,
        task: str | None = None,
        dataset: str | None = None,
        metric: str | None = None,
        status: str | None = None,
    ) -> tuple[CatalogExperiment, ...]:
        conditions = []
        values: list[Any] = []
        for column, value in [
            ("artifact_type", artifact_type),
            ("task", task),
            ("dataset_identity", dataset),
            ("primary_metric", metric),
        ]:
            if value is not None:
                conditions.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.connection.execute(
            "SELECT id FROM experiments" + where + " ORDER BY id", values
        )
        experiments = tuple(self.show(row[0]) for row in rows)
        return tuple(
            item
            for item in experiments
            if status is None
            or any(location.descriptor.status == status for location in item.locations)
        )

    @_database_operation
    def verify(self, identifier: str, *, deep: bool = False) -> tuple[VerificationReport, ...]:
        experiment = self.show(identifier)
        reports = []
        for location in experiment.locations:
            report = verify(location.directory, deep=deep)
            changed = False
            snapshot_error = None
            if report.passed:
                try:
                    changed = _evidence_digest(location.directory) != location.evidence_digest
                except (OSError, ArtifactError) as exc:
                    snapshot_error = str(exc)
            if changed or snapshot_error is not None:
                from benchforge.artifacts.types import Diagnostic

                report = report.model_copy(
                    update={
                        "diagnostics": (
                            *report.diagnostics,
                            Diagnostic(
                                level="FAIL",
                                code="indexed_evidence_changed",
                                message=snapshot_error or "artifact bytes changed since ingestion",
                            ),
                        ),
                        "integrity": "invalid",
                    }
                )
            reports.append(report)
        with self.connection:
            for location, report in zip(experiment.locations, reports, strict=True):
                row = self.connection.execute(
                    "SELECT id FROM artifact_locations WHERE directory=?",
                    (str(location.directory),),
                ).fetchone()
                self._record_verification(row[0], report)
        return tuple(reports)
