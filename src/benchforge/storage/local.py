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
from benchforge.core.config import RunConfig

if TYPE_CHECKING:
    from benchforge.execution.runner import RunResult


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: RunConfig, result: RunResult) -> Path:
        timestamp = datetime.now(UTC)
        run_name = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{result.fingerprint[:12]}"
        directory = self.root / run_name
        return self.write_at(directory, config, result, created_at=timestamp)

    def write_at(
        self,
        directory: Path,
        config: RunConfig,
        result: RunResult,
        *,
        created_at: datetime | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> Path:
        timestamp = created_at or datetime.now(UTC)
        directory.mkdir(parents=True, exist_ok=False)
        self._write_json(directory / "config.json", config.canonical_dict())
        metadata = {
            "created_at": timestamp.isoformat(),
            "benchforge_version": __version__,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "dataset_identity": result.dataset_identity,
            "dataset_summary": asdict(result.dataset),
            "config_fingerprint": result.fingerprint,
            "fingerprint_method": "resolved-run-v1",
            "seed": result.seed,
        }
        metadata.update(metadata_extra or {})
        self._write_json(directory / "metadata.json", metadata)
        self._write_json(
            directory / "fold_metrics.json",
            [asdict(fold) for fold in result.fold_results],
        )
        self._write_json(
            directory / "aggregate_metrics.json",
            {name: asdict(metric) for name, metric in result.aggregate_metrics.items()},
        )
        with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_index", "fold_id", "truth", "prediction", "score"]
            )
            writer.writeheader()
            writer.writerows(asdict(prediction) for prediction in result.predictions)
        seal(directory)
        return directory

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_run_artifacts(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    with (path / "predictions.csv").open(encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    return {
        "config": json.loads((path / "config.json").read_text(encoding="utf-8")),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
        "fold_metrics": json.loads((path / "fold_metrics.json").read_text(encoding="utf-8")),
        "aggregate_metrics": json.loads(
            (path / "aggregate_metrics.json").read_text(encoding="utf-8")
        ),
        "predictions": predictions,
    }
