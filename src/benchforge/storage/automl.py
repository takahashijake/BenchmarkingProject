from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import sklearn

from benchforge._version import __version__
from benchforge.artifacts.manifest import seal
from benchforge.automl.results import AutoMLResult
from benchforge.core.config import AutoMLConfig
from benchforge.storage.search import SearchArtifactStore


class AutoMLArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: AutoMLConfig, result: AutoMLResult) -> tuple[Path, Path]:
        timestamp = datetime.now(UTC)
        name = f"automl_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{result.fingerprint[:12]}"
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=False)
        search_directory = directory / "search"
        SearchArtifactStore(search_directory).write_at(
            search_directory,
            result.plan.search_config,
            result.search_result,
            created_at=timestamp,
        )

        _write_json(directory / "automl_config.json", config.canonical_dict())
        _write_json(directory / "plan.json", result.plan.canonical_dict())
        _write_json(
            directory / "generated_search_config.json",
            result.plan.search_config.canonical_dict(),
        )
        _write_json(
            directory / "metadata.json",
            {
                "created_at": timestamp.isoformat(),
                "benchforge_version": __version__,
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "optuna_version": optuna.__version__,
                "automl_fingerprint": result.fingerprint,
                "selected_model_identity": result.plan.selected_model_identity,
                "fingerprint_method": "canonical-metrics-v1",
                "search_fingerprint": result.search_result.fingerprint,
                "dataset_identity": result.plan.dataset_identity,
                "selected_searchable_families": result.plan.searchable_families,
                "selected_fixed_candidates": result.plan.fixed_families,
                "total_trial_budget": result.plan.total_trial_budget,
                "allocated_trials": result.plan.allocated_trials,
                "allocated_outer_trials": result.plan.allocated_outer_trials,
                "unallocated_trials": result.plan.unallocated_trials,
                "final_search_trials": result.plan.final_search_trials,
                "approximate_maximum_fits": result.plan.approximate_maximum_fits,
                "seed": config.seed,
                "total_duration_seconds": result.total_duration_seconds,
            },
        )
        _write_json(
            directory / "result.json",
            {
                "automl_fingerprint": result.fingerprint,
                "search_fingerprint": result.search_result.fingerprint,
                "primary_metric": result.search_result.primary_metric,
                "optimization_direction": result.search_result.optimization_direction,
                "top_measured_family": next(
                    entry.model_identifier
                    for entry in result.search_result.leaderboard
                    if entry.status == "success"
                ),
                "leaderboard": [asdict(entry) for entry in result.search_result.leaderboard],
                "final_candidate": (
                    None
                    if result.search_result.final_candidate is None
                    else asdict(result.search_result.final_candidate)
                ),
                "search_artifacts": "search",
            },
        )
        seal(directory)
        return directory, search_directory


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_automl_artifacts(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    return {
        "automl_config": json.loads((path / "automl_config.json").read_text(encoding="utf-8")),
        "plan": json.loads((path / "plan.json").read_text(encoding="utf-8")),
        "generated_search_config": json.loads(
            (path / "generated_search_config.json").read_text(encoding="utf-8")
        ),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
        "result": json.loads((path / "result.json").read_text(encoding="utf-8")),
    }
