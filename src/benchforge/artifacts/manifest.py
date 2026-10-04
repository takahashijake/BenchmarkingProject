"""Versioned byte seals, separate from semantic experiment fingerprints."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from benchforge.artifacts.discovery import inventory
from benchforge.artifacts.types import ArtifactType, EvidenceModel


class ManifestMember(EvidenceModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_path(self) -> "ManifestMember":
        parts = self.path.split("/")
        if (
            any(part in {"", ".", ".."} for part in parts)
            or "\\" in self.path
            or "\x00" in self.path
            or ":" in parts[0]
        ):
            raise ValueError("manifest member must be a canonical relative path")
        if self.path == "manifest.json":
            raise ValueError("manifest cannot hash itself")
        return self


class Manifest(EvidenceModel):
    manifest_schema_version: Literal[1] = 1
    artifact_type: ArtifactType
    experiment_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchforge_version: str | None
    created_at: str | None
    protection_policy: Literal["all-regular-files-except-root-manifest-v1"] = (
        "all-regular-files-except-root-manifest-v1"
    )
    files: tuple[ManifestMember, ...]

    @model_validator(mode="after")
    def ordered_inventory(self) -> "Manifest":
        paths = [item.path for item in self.files]
        if not paths or paths != sorted(set(paths)):
            raise ValueError("manifest inventory must be nonempty, sorted and unique")
        return self


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal(directory: Path) -> None:
    """Validate evidence first and publish the manifest last, without overwriting a seal."""
    from benchforge.artifacts.verify import verify

    if (directory / "manifest.json").exists():
        raise ValueError(f"artifact already sealed: {directory}")
    report = verify(directory, _allow_unsealed=True)
    if not report.passed or report.descriptor is None:
        details = "; ".join(item.message for item in report.diagnostics if item.level == "FAIL")
        raise ValueError(f"cannot seal incomplete artifact {directory}: {details}")
    descriptor = report.descriptor
    manifest = Manifest(
        artifact_type=descriptor.artifact_type,
        experiment_fingerprint=descriptor.fingerprint,
        benchforge_version=descriptor.benchforge_version,
        created_at=descriptor.created_at,
        files=tuple(
            ManifestMember(path=name, sha256=digest(directory / name))
            for name in inventory(directory)
        ),
    )
    # Exclusive creation prevents accidental re-sealing. An interrupted write is an invalid seal.
    with (directory / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
