"""Structural classification and deterministic, symlink-free traversal."""

import os
from pathlib import Path

from benchforge.artifacts.types import ArtifactError, ArtifactType, Diagnostic, EvidenceModel

CONFIG_FILES = {
    ArtifactType.RUN: "config.json",
    ArtifactType.BENCHMARK: "benchmark_config.json",
    ArtifactType.SEARCH: "search_config.json",
    ArtifactType.AUTOML: "automl_config.json",
    ArtifactType.ROBUSTNESS: "robustness_config.json",
}


def classify(directory: Path) -> ArtifactType:
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactError(f"unsupported artifact directory: {directory}")
    matches = [kind for kind, filename in CONFIG_FILES.items() if (directory / filename).exists()]
    if len(matches) != 1:
        label = "ambiguous" if matches else "unsupported"
        raise ArtifactError(f"{label} artifact: {directory}; signatures: {matches}")
    return matches[0]


class DiscoveryResult(EvidenceModel):
    directories: tuple[Path, ...]
    diagnostics: tuple[Diagnostic, ...]


def discover(root: str | Path, *, validate: bool = True) -> DiscoveryResult:
    """Return top-level experiments; composed child evidence belongs to its parent."""
    path = Path(root).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ArtifactError(f"discovery root is not a regular directory: {path}")
    path = path.resolve()
    found: list[Path] = []
    diagnostics: list[Diagnostic] = []

    def visit(directory: Path) -> None:
        names = {entry.name for entry in directory.iterdir()}
        evidence_names = {
            "aggregate_metrics.json",
            "leaderboard.json",
            "tuned_leaderboard.json",
            "robustness_leaderboard.json",
            "repetition_plan.json",
            "plan.json",
            "result.json",
        }
        if (
            names.intersection(CONFIG_FILES.values())
            or "manifest.json" in names
            or ("metadata.json" in names and names.intersection(evidence_names))
        ):
            try:
                classify(directory)
            except ArtifactError as exc:
                diagnostics.append(
                    Diagnostic(level="FAIL", code="classification", message=str(exc))
                )
            else:
                found.append(directory)
                if validate:
                    from benchforge.artifacts.verify import verify

                    report = verify(directory)
                    diagnostics.extend(report.diagnostics)
            return
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                diagnostics.append(
                    Diagnostic(
                        level="WARNING", code="symlink_skipped", message=f"skipped symlink {entry}"
                    )
                )
            elif entry.is_dir():
                visit(entry)

    try:
        visit(path)
    except OSError as exc:
        raise ArtifactError(f"cannot discover {path}: {exc}") from exc
    return DiscoveryResult(directories=tuple(found), diagnostics=tuple(diagnostics))


def inventory(directory: Path) -> tuple[str, ...]:
    """Protect every regular file, including child manifests; exclude only the root seal."""
    files: list[str] = []
    for base, folders, names in os.walk(directory, followlinks=False):
        folders.sort()
        for name in sorted([*folders, *names]):
            path = Path(base) / name
            if path.is_symlink():
                raise ArtifactError(f"symlink forbidden inside artifact: {path}")
            if path.is_file():
                relative = path.relative_to(directory).as_posix()
                if relative != "manifest.json":
                    files.append(relative)
            elif not path.is_dir():
                raise ArtifactError(f"non-regular artifact member: {path}")
    return tuple(sorted(files))
