from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchforge.automl.planner import AutoMLPlan
from benchforge.search.results import SearchResult


@dataclass(frozen=True)
class AutoMLResult:
    fingerprint: str
    plan: AutoMLPlan
    search_result: SearchResult
    total_duration_seconds: float
    artifact_directory: Path | None = None
