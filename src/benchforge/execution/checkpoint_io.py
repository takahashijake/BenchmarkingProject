"""Safe structured checkpoint bytes and crash-durable publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchforge.providers.json import unique_object


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    """Publish fully flushed bytes via same-directory rename, never partial JSON."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_regular_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing checkpoint file: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
