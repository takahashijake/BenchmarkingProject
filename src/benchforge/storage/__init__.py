"""Local machine-readable run artifacts."""

from typing import Any

from benchforge.storage.local import LocalArtifactStore, read_run_artifacts
from benchforge.storage.suite import BenchmarkArtifactStore, read_benchmark_artifacts

__all__ = [
    "BenchmarkArtifactStore",
    "LocalArtifactStore",
    "SearchArtifactStore",
    "read_benchmark_artifacts",
    "read_run_artifacts",
    "read_search_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in {"SearchArtifactStore", "read_search_artifacts"}:
        from benchforge.storage.search import SearchArtifactStore, read_search_artifacts

        return {
            "SearchArtifactStore": SearchArtifactStore,
            "read_search_artifacts": read_search_artifacts,
        }[name]
    raise AttributeError(name)
