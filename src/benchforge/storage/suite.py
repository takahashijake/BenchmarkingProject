from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sklearn

from benchforge._version import __version__
from benchforge.artifacts.manifest import seal
from benchforge.core.config import BenchmarkConfig
from benchforge.storage.local import LocalArtifactStore

if TYPE_CHECKING:
    from benchforge.execution.benchmark import BenchmarkResult


class BenchmarkArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: BenchmarkConfig, result: BenchmarkResult) -> Path:
        timestamp = datetime.now(UTC)
        name = f"benchmark_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{result.fingerprint[:12]}"
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "benchmark_config.json", config.canonical_dict())
        _write_json(
            directory / "metadata.json",
            {
                "created_at": timestamp.isoformat(),
                "benchforge_version": __version__,
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "benchmark_fingerprint": result.fingerprint,
                "fingerprint_method": "canonical-metrics-v1",
                "dataset_identity": result.dataset.identity,
                "seed": config.seed,
                "primary_metric": result.primary_metric,
                "fold_count": result.fold_count,
                "successful_candidates": len(result.candidates),
                "failed_candidates": len(result.failures),
                "total_duration_seconds": result.total_duration_seconds,
            },
        )
        _write_json(directory / "dataset_summary.json", asdict(result.dataset))
        _write_json(directory / "leaderboard.json", [asdict(row) for row in result.leaderboard])
        _write_leaderboard_csv(directory / "leaderboard.csv", config, result)
        _write_json(directory / "failures.json", [asdict(failure) for failure in result.failures])

        runs_directory = directory / "runs"
        runs_directory.mkdir()
        run_store = LocalArtifactStore(runs_directory)
        for candidate in result.candidates:
            run_store.write_at(
                runs_directory / candidate.model_identifier,
                config.to_run_config(candidate.model_config),
                candidate.run_result,
                created_at=timestamp,
                metadata_extra={
                    "model_identifier": candidate.model_identifier,
                    "duration_seconds": candidate.duration_seconds,
                    "benchmark_fingerprint": result.fingerprint,
                },
            )
        seal(directory)
        return directory


def _write_leaderboard_csv(path: Path, config: BenchmarkConfig, result: BenchmarkResult) -> None:
    metric_fields = [
        field
        for metric in config.metrics
        for field in (f"{metric.value}_mean", f"{metric.value}_std")
    ]
    fields = [
        "rank",
        "model_identifier",
        "status",
        "primary_metric_mean",
        "primary_metric_std",
        "duration_seconds",
        *metric_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in result.leaderboard:
            row: dict[str, Any] = {
                "rank": entry.rank,
                "model_identifier": entry.model_identifier,
                "status": entry.status,
                "primary_metric_mean": entry.primary_metric_mean,
                "primary_metric_std": entry.primary_metric_std,
                "duration_seconds": entry.duration_seconds,
            }
            for name, metric in entry.metrics.items():
                row[f"{name}_mean"] = metric.mean
                row[f"{name}_std"] = metric.std
            writer.writerow(row)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_benchmark_artifacts(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    with (path / "leaderboard.csv").open(encoding="utf-8") as handle:
        leaderboard_csv = list(csv.DictReader(handle))
    return {
        "benchmark_config": json.loads(
            (path / "benchmark_config.json").read_text(encoding="utf-8")
        ),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
        "dataset_summary": json.loads((path / "dataset_summary.json").read_text(encoding="utf-8")),
        "leaderboard": json.loads((path / "leaderboard.json").read_text(encoding="utf-8")),
        "leaderboard_csv": leaderboard_csv,
        "failures": json.loads((path / "failures.json").read_text(encoding="utf-8")),
    }
