from __future__ import annotations

import csv
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
from benchforge.core.config import SearchConfig
from benchforge.search.results import SearchResult, TrialRecord


class SearchArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, config: SearchConfig, result: SearchResult) -> Path:
        timestamp = datetime.now(UTC)
        name = f"search_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{result.fingerprint[:12]}"
        directory = self.root / name
        return self.write_at(directory, config, result, created_at=timestamp)

    def write_at(
        self,
        directory: Path,
        config: SearchConfig,
        result: SearchResult,
        *,
        created_at: datetime | None = None,
    ) -> Path:
        timestamp = created_at or datetime.now(UTC)
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "search_config.json", config.canonical_dict())
        _write_json(
            directory / "metadata.json",
            {
                "created_at": timestamp.isoformat(),
                "benchforge_version": __version__,
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "optuna_version": optuna.__version__,
                "search_fingerprint": result.fingerprint,
                "search_space_identity": result.search_space_identity,
                "fingerprint_method": "canonical-metrics-v1",
                "dataset_identity": result.dataset.identity,
                "seed": config.seed,
                "primary_metric": result.primary_metric,
                "optimization_direction": result.optimization_direction,
                "outer_fold_count": result.outer_fold_count,
                "inner_fold_count": result.inner_fold_count,
                "successful_families": len(result.families),
                "failed_families": len(result.failures),
                "total_duration_seconds": result.total_duration_seconds,
            },
        )
        _write_json(directory / "dataset_summary.json", asdict(result.dataset))
        _write_json(
            directory / "tuned_leaderboard.json",
            [asdict(entry) for entry in result.leaderboard],
        )
        _write_leaderboard_csv(directory / "tuned_leaderboard.csv", config, result)
        _write_json(directory / "failures.json", [asdict(item) for item in result.failures])
        _write_json(
            directory / "final_candidate.json",
            None if result.final_candidate is None else asdict(result.final_candidate),
        )
        _write_json(
            directory / "outer_folds.json",
            [
                {
                    "fold_id": fold.fold_id,
                    "train_indices": fold.train_indices.tolist(),
                    "validation_indices": fold.validation_indices.tolist(),
                }
                for fold in result.outer_folds
            ],
        )

        families_directory = directory / "families"
        families_directory.mkdir()
        for family in result.families:
            family_directory = families_directory / family.model_identifier
            family_directory.mkdir()
            _write_json(
                family_directory / "summary.json",
                {
                    "model_identifier": family.model_identifier,
                    "candidate": family.candidate.model_dump(mode="json", exclude_none=True),
                    "aggregate_metrics": {
                        name: asdict(metric) for name, metric in family.aggregate_metrics.items()
                    },
                    "duration_seconds": family.duration_seconds,
                    "configured_trials": family.configured_trials,
                    "completed_trials": family.completed_trials,
                    "failed_trials": family.failed_trials,
                    "pruned_trials": family.pruned_trials,
                },
            )
            _write_predictions(family_directory / "outer_predictions.csv", family.predictions)
            for fold in family.outer_folds:
                fold_directory = family_directory / f"outer_fold_{fold.fold_id}"
                fold_directory.mkdir()
                _write_json(
                    fold_directory / "split_plan.json",
                    {
                        "outer_train_indices": fold.outer_train_indices,
                        "outer_validation_indices": fold.outer_validation_indices,
                        "inner_folds": [
                            {
                                "fold_id": inner.fold_id,
                                "train_indices": inner.train_indices.tolist(),
                                "validation_indices": inner.validation_indices.tolist(),
                            }
                            for inner in fold.inner_folds
                        ],
                    },
                )
                _write_json(
                    fold_directory / "trials.json", [asdict(trial) for trial in fold.trials]
                )
                _write_trials_csv(fold_directory / "trials.csv", fold.trials)
                _write_json(fold_directory / "best_params.json", fold.selected_parameters)
                _write_json(
                    fold_directory / "outer_result.json",
                    {
                        "fold_result": asdict(fold.outer_result),
                        "best_inner_score": fold.best_inner_score,
                        "duration_seconds": fold.duration_seconds,
                        "configured_trials": fold.configured_trials,
                        "completed_trials": fold.completed_trials,
                        "failed_trials": fold.failed_trials,
                        "pruned_trials": fold.pruned_trials,
                        "sampler_seed": fold.sampler_seed,
                    },
                )
        for failure in result.failures:
            family_directory = families_directory / failure.model_identifier
            family_directory.mkdir()
            _write_json(family_directory / "failure.json", asdict(failure))
            _write_json(
                family_directory / "trials.json",
                [asdict(trial) for trial in failure.trials],
            )
            _write_trials_csv(family_directory / "trials.csv", failure.trials)
        seal(directory)
        return directory


def _write_leaderboard_csv(path: Path, config: SearchConfig, result: SearchResult) -> None:
    metric_fields = [
        field
        for metric in config.metrics
        for field in (f"{metric.value}_mean", f"{metric.value}_std")
    ]
    fields = [
        "rank",
        "model_identifier",
        "mode",
        "status",
        "primary_metric_mean",
        "primary_metric_std",
        "configured_trials",
        "completed_trials",
        "failed_trials",
        "pruned_trials",
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
                "mode": entry.mode,
                "status": entry.status,
                "primary_metric_mean": entry.primary_metric_mean,
                "primary_metric_std": entry.primary_metric_std,
                "configured_trials": entry.configured_trials,
                "completed_trials": entry.completed_trials,
                "failed_trials": entry.failed_trials,
                "pruned_trials": entry.pruned_trials,
                "duration_seconds": entry.duration_seconds,
            }
            for name, metric in entry.metrics.items():
                row[f"{name}_mean"] = metric.mean
                row[f"{name}_std"] = metric.std
            writer.writerow(row)


def _write_trials_csv(path: Path, trials: tuple[TrialRecord, ...]) -> None:
    fields = [
        "number",
        "state",
        "value",
        "duration_seconds",
        "proposed_parameters",
        "parameters",
        "error_type",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            row = asdict(trial)
            row["proposed_parameters"] = json.dumps(
                row["proposed_parameters"], sort_keys=True, separators=(",", ":")
            )
            row["parameters"] = json.dumps(row["parameters"], sort_keys=True, separators=(",", ":"))
            writer.writerow(row)


def _write_predictions(path: Path, predictions: tuple[Any, ...]) -> None:
    fields = ["sample_index", "fold_id", "truth", "prediction", "score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(prediction) for prediction in predictions)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_search_artifacts(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    with (path / "tuned_leaderboard.csv").open(encoding="utf-8") as handle:
        leaderboard_csv = list(csv.DictReader(handle))
    return {
        "search_config": json.loads((path / "search_config.json").read_text(encoding="utf-8")),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
        "dataset_summary": json.loads((path / "dataset_summary.json").read_text(encoding="utf-8")),
        "tuned_leaderboard": json.loads(
            (path / "tuned_leaderboard.json").read_text(encoding="utf-8")
        ),
        "tuned_leaderboard_csv": leaderboard_csv,
        "failures": json.loads((path / "failures.json").read_text(encoding="utf-8")),
        "final_candidate": json.loads((path / "final_candidate.json").read_text(encoding="utf-8")),
        "outer_folds": json.loads((path / "outer_folds.json").read_text(encoding="utf-8")),
    }
