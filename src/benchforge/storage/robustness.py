"""Versioned, auditable robustness artifacts, composing existing artifact helpers."""

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchforge.robustness.config import RobustnessConfig
from benchforge.robustness.planner import dependency_versions
from benchforge.robustness.results import RobustnessResult
from benchforge.storage.local import LocalArtifactStore


class RobustnessArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: RobustnessConfig, result: RobustnessResult) -> Path:
        timestamp = datetime.now(UTC)
        name = f"robustness_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{result.fingerprint[:12]}"
        return self.write_at(self.root / name, config, result, created_at=timestamp)

    def write_at(
        self,
        directory: Path,
        config: RobustnessConfig,
        result: RobustnessResult,
        *,
        created_at: datetime | None = None,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=False)
        write_json = LocalArtifactStore._write_json
        write_json(directory / "robustness_config.json", config.canonical_dict())
        write_json(
            directory / "metadata.json",
            {
                "artifact_schema_version": 1,
                "created_at": (created_at or datetime.now(UTC)).isoformat(),
                "versions": dependency_versions(),
                "robustness_fingerprint": result.fingerprint,
                "dataset_identity": result.dataset.identity,
                "primary_metric": result.primary_metric,
                "optimization_direction": result.optimization_direction,
                "methodology_version": result.plan.methodology_version,
                "rank_policy": "exact ties share competition ranks (1, 1, 3)",
                "rank_rate_denominator": "all configured repetitions, including failures",
                "leaderboard_policy": "complete before partial before failed; then metric mean, id",
                "standard_deviation": "sample (ddof=1); single observation descriptive std=0",
                "interval_interpretation": "resampling sensitivity conditional on this dataset",
                "total_duration_seconds": result.total_duration_seconds,
            },
        )
        write_json(directory / "dataset_summary.json", asdict(result.dataset))
        write_json(
            directory / "repetition_plan.json",
            {
                "candidate_identifiers": result.plan.candidate_identifiers,
                "approximate_maximum_fits": result.plan.approximate_maximum_fits,
                "repetitions": [
                    {"repetition_id": item.repetition_id, "seed": item.seed}
                    for item in result.plan.repetitions
                ],
            },
        )
        rows = [asdict(entry) for entry in result.leaderboard]
        write_json(directory / "robustness_leaderboard.json", rows)
        with (directory / "robustness_leaderboard.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = [key for key in rows[0] if key != "metrics"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        write_json(
            directory / "pairwise_comparisons.json",
            [asdict(pair) for pair in result.pairwise_comparisons],
        )
        write_json(
            directory / "rank_stability.json",
            [
                {
                    **asdict(item),
                    "rank_counts": {str(row.rank): row.count for row in item.rank_counts},
                }
                for item in result.rank_stability
            ],
        )
        write_json(directory / "failures.json", [asdict(item) for item in result.failures])
        for plan, repetition in zip(result.plan.repetitions, result.repetitions, strict=True):
            path = directory / "repetitions" / f"repetition_{plan.repetition_id:03d}"
            path.mkdir(parents=True)
            write_json(
                path / "split_plan.json",
                [
                    {
                        "fold_id": fold.fold_id,
                        "train_indices": fold.train_indices.tolist(),
                        "validation_indices": fold.validation_indices.tolist(),
                    }
                    for fold in plan.folds
                ],
            )
            write_json(path / "ranks.json", [asdict(row) for row in repetition.ranks])
            write_json(
                path / "metrics.json",
                [
                    {
                        "model_identifier": candidate.model_identifier,
                        "status": candidate.status,
                        "metrics": [asdict(metric) for metric in candidate.metrics],
                    }
                    for candidate in repetition.candidates
                ],
            )
            for candidate in repetition.candidates:
                candidate_path = path / "candidates" / candidate.model_identifier
                candidate_path.mkdir(parents=True)
                summary = asdict(candidate)
                summary.pop("predictions")
                write_json(candidate_path / "summary.json", summary)
                with (candidate_path / "predictions.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["sample_index", "fold_id", "truth", "prediction", "score"],
                    )
                    writer.writeheader()
                    writer.writerows(asdict(item) for item in candidate.predictions)
        return directory


def read_robustness_artifacts(directory: str | Path) -> dict[str, Any]:
    """Read all semantic evidence, including partial predictions and failed summaries."""
    path = Path(directory)

    def read_json(file: Path) -> Any:
        return json.loads(file.read_text(encoding="utf-8"))

    def read_csv(file: Path) -> list[dict[str, Any]]:
        with file.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    result = {
        name: read_json(path / f"{name}.json")
        for name in (
            "robustness_config",
            "metadata",
            "dataset_summary",
            "repetition_plan",
            "robustness_leaderboard",
            "pairwise_comparisons",
            "rank_stability",
            "failures",
        )
    }
    result["robustness_leaderboard_csv"] = read_csv(path / "robustness_leaderboard.csv")
    repetitions = []
    for plan in result["repetition_plan"]["repetitions"]:
        repetition = path / "repetitions" / f"repetition_{plan['repetition_id']:03d}"
        repetitions.append(
            {
                **plan,
                "split_plan": read_json(repetition / "split_plan.json"),
                "ranks": read_json(repetition / "ranks.json"),
                "metrics": read_json(repetition / "metrics.json"),
                "candidates": {
                    identifier: {
                        "summary": read_json(
                            repetition / "candidates" / identifier / "summary.json"
                        ),
                        "predictions": read_csv(
                            repetition / "candidates" / identifier / "predictions.csv"
                        ),
                    }
                    for identifier in result["repetition_plan"]["candidate_identifiers"]
                },
            }
        )
    result["repetitions"] = repetitions
    return result
