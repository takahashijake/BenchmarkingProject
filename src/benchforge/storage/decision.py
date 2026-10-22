"""Safely staged, manifest-sealed publication of decision contract 1 evidence."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from benchforge.artifacts.discovery import CONFIG_FILES
from benchforge.artifacts.manifest import seal
from benchforge.decision.config import DecisionConfig
from benchforge.decision.planning import METHODOLOGY_VERSION, dataset_content_identity
from benchforge.decision.results import DecisionResult
from benchforge.decision.verification import validate_decision
from benchforge.execution.checkpoint_io import _write_atomic


class DecisionArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: DecisionConfig, result: DecisionResult) -> Path:
        """Expose an experiment only after validation and sealing; never overwrite."""
        identity = validate_decision(config, result)
        root = self.root.absolute()
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("decision artifact root must not use symlinks")
        for parent in (root, *root.parents):
            if any(
                (parent / marker).exists()
                for marker in (
                    "manifest.json",
                    "workspace.json",
                    *CONFIG_FILES.values(),
                )
            ):
                raise ValueError("decision output must be outside existing evidence/workspaces")
        root.mkdir(parents=True, exist_ok=True)
        created = datetime.now(UTC)
        name = (
            f"decision_{created.strftime('%Y%m%dT%H%M%S.%fZ')}_"
            f"{result.fingerprint[:12]}_{uuid4().hex}"
        )
        destination = root / name
        staging = Path(tempfile.mkdtemp(prefix=".decision-stage-", dir=root))
        try:
            _write_atomic(staging / "decision_config.json", config.canonical_dict())
            _write_atomic(staging / "decision_result.json", result.model_dump(mode="json"))
            _write_atomic(
                staging / "metadata.json",
                {
                    "artifact_schema_version": 1,
                    "benchforge_version": identity["versions"]["benchforge"],
                    "created_at": created.isoformat(),
                    "decision_fingerprint": result.fingerprint,
                    "dataset_identity": dataset_content_identity(result.dataset),
                    "methodology_version": METHODOLOGY_VERSION,
                    "versions": identity["versions"],
                },
            )
            seal(staging)
            with (staging / "manifest.json").open("r+b") as handle:
                os.fsync(handle.fileno())
            # Directory rename requires a nonexistent destination on Windows.
            # Serialize cooperating publishers, reject collisions, then rename the
            # complete sealed directory on the same filesystem. The small lock
            # file is publication metadata, not experiment evidence.
            lock = root / ".decision-publication.lock"
            if lock.is_symlink():
                raise ValueError("decision publication lock must not be a symlink")
            with FileLock(str(lock)):
                if destination.exists():
                    raise FileExistsError(f"decision destination already exists: {destination}")
                os.rename(staging, destination)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)
