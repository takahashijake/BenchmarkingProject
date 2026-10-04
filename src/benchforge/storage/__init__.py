"""Local machine-readable run artifacts."""

from typing import Any

from benchforge.storage.local import LocalArtifactStore, read_run_artifacts
from benchforge.storage.suite import BenchmarkArtifactStore, read_benchmark_artifacts

__all__ = [
    "BenchmarkArtifactStore",
    "AutoMLArtifactStore",
    "RobustnessArtifactStore",
    "read_robustness_artifacts",
    "LocalArtifactStore",
    "SearchArtifactStore",
    "read_benchmark_artifacts",
    "read_automl_artifacts",
    "read_run_artifacts",
    "read_search_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in {"RobustnessArtifactStore", "read_robustness_artifacts"}:
        from benchforge.storage.robustness import RobustnessArtifactStore, read_robustness_artifacts

        return {
            "RobustnessArtifactStore": RobustnessArtifactStore,
            "read_robustness_artifacts": read_robustness_artifacts,
        }[name]
    if name in {"AutoMLArtifactStore", "read_automl_artifacts"}:
        from benchforge.storage.automl import AutoMLArtifactStore, read_automl_artifacts

        return {
            "AutoMLArtifactStore": AutoMLArtifactStore,
            "read_automl_artifacts": read_automl_artifacts,
        }[name]
    if name in {"SearchArtifactStore", "read_search_artifacts"}:
        from benchforge.storage.search import SearchArtifactStore, read_search_artifacts

        return {
            "SearchArtifactStore": SearchArtifactStore,
            "read_search_artifacts": read_search_artifacts,
        }[name]
    raise AttributeError(name)
