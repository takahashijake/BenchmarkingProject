"""Validated, atomic HTML publication outside artifact and recovery trees."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from benchforge.artifacts.discovery import CONFIG_FILES
from benchforge.reporting.loading import load_comparison, load_report, regular_path
from benchforge.reporting.models import ReportingError
from benchforge.reporting.rendering import render_comparison, render_report


def _destination(output: str | Path, sources: tuple[Path, ...], overwrite: bool) -> Path:
    path = regular_path(output)
    if not path.parent.is_dir():
        raise ReportingError(
            f"output parent directory does not exist: {path.parent}; create it first"
        )
    for source in sources:
        if path.is_relative_to(source):
            raise ReportingError(f"output must be outside the source artifact tree: {path}")
    for parent in path.parents:
        if any(
            (parent / marker).exists()
            for marker in ("manifest.json", "workspace.json", *CONFIG_FILES.values())
        ):
            raise ReportingError(f"output is inside an artifact or recovery workspace: {parent}")
    if path.exists():
        if not path.is_file():
            raise ReportingError(f"output is not a regular file: {path}")
        if not overwrite:
            raise ReportingError(
                f"output already exists: {path}; use overwrite=True or --overwrite"
            )
    return path


def _publish(path: Path, content: str, *, overwrite: bool) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".benchforge-report-", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication; never expose a partially written destination.
            # If hard links are unsupported, fail explicitly instead of unsafe check-then-rename.
            os.link(temporary, path)
    except OSError as exc:
        raise ReportingError(
            f"cannot publish report to {path}: {exc}; check permissions, "
            "free space and whether the destination already exists"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_report(
    directory: str | Path,
    output: str | Path,
    *,
    compare: str | Path | None = None,
    deep: bool = False,
    allow_legacy: bool = False,
    overwrite: bool = False,
) -> Path:
    """Verify, normalize, render and atomically publish an offline HTML report.

    Only pre-v0.8 unsealed evidence accepted by the existing verifier can use
    ``allow_legacy``. Missing modern seals and all verification failures stay errors.
    Parent output directories must already exist. Source evidence is never written.
    """
    try:
        sources = (regular_path(directory),) + (
            (regular_path(compare),) if compare is not None else ()
        )
        path = _destination(output, sources, overwrite)
        if compare is None:
            content = render_report(load_report(directory, deep=deep, allow_legacy=allow_legacy))
        else:
            content = render_comparison(
                load_comparison(directory, compare, deep=deep, allow_legacy=allow_legacy)
            )
        _destination(output, sources, overwrite)
        _publish(path, content, overwrite=overwrite)
        return path
    except ReportingError:
        raise
    except (OSError, ValueError) as exc:
        raise ReportingError(f"cannot generate report: {exc}") from exc
