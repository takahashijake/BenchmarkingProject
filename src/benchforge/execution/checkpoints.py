"""Versioned, atomic candidate checkpoints for the fixed benchmark workflow.

Workspaces are deliberately separate from the sealed experiment artifact schema.
They contain only recomputable results, never executable/pickled model state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sklearn
from filelock import FileLock, Timeout
from pydantic import TypeAdapter

from benchforge._version import __version__
from benchforge.core.config import BenchmarkConfig, ModelConfig
from benchforge.data.registry import Dataset
from benchforge.evaluation.metrics import aggregate_fold_metrics
from benchforge.execution.runner import RunResult
from benchforge.splits.stratified import Fold

if TYPE_CHECKING:
    from benchforge.execution.benchmark import CandidateResult

# Increment this whenever evaluation or checkpoint compatibility semantics change.
WORKSPACE_SCHEMA = 1
EXECUTION_CONTRACT = "fixed-benchmark-candidate-v1"
_RUN_ADAPTER: TypeAdapter[RunResult] = TypeAdapter(RunResult)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _dataset_digest(dataset: Dataset) -> str:
    """Hash resolved values, ordering, index, schema, and target, not a source path."""
    digest = hashlib.sha256()
    digest.update(_canonical({
        "features": [(str(name), str(dtype)) for name, dtype in dataset.features.dtypes.items()],
        "target_dtype": str(dataset.target.dtype),
        "task": dataset.task.value,
        "numeric": dataset.numeric_feature_names,
        "categorical": dataset.categorical_feature_names,
    }))
    digest.update(pd.util.hash_pandas_object(dataset.features, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(dataset.target, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def experiment_identity(
    config: BenchmarkConfig, dataset: Dataset, folds: tuple[Fold, ...]
) -> str:
    fold_plan = [
        {
            "id": fold.fold_id,
            "train": fold.train_indices.tolist(),
            "validation": fold.validation_indices.tolist(),
        }
        for fold in folds
    ]
    return _sha({
        "schema": WORKSPACE_SCHEMA,
        "execution": EXECUTION_CONTRACT,
        "benchmark": config.fingerprint_for_dataset(dataset.identity),
        "dataset_digest": _dataset_digest(dataset),
        "folds": fold_plan,
        "benchforge": __version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })


def _write_atomic(path: Path, value: Any) -> None:
    """Publish fully flushed bytes via same-directory rename, never partial JSON."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
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
    return json.loads(path.read_text(encoding="utf-8"))


class CheckpointWorkspace:
    """Own one exclusive workspace lock for the entire run.

    FileLock uses OS-managed locks, so a killed process cannot strand ownership.
    A completed checkpoint is a validated immutable candidate result, not a
    promise that a process merely began a task.
    """

    def __init__(
        self,
        path: Path,
        config: BenchmarkConfig,
        dataset: Dataset,
        folds: tuple[Fold, ...],
        *,
        resume: bool,
    ) -> None:
        self.path = path.absolute()
        self.config = config
        self.dataset = dataset
        self.folds = folds
        self.resume = resume
        self.identity = experiment_identity(config, dataset, folds)
        self.task_keys = {
            model.id or model.name: _sha({
                "experiment": self.identity,
                "model": model.model_dump(mode="json", exclude_none=True),
            })
            for model in config.models
        }
        self.lock = FileLock(str(self.path) + ".lock", timeout=0)
        self.reused: list[str] = []
        self.executed: list[str] = []
        self.invalid: list[str] = []

    def __enter__(self) -> CheckpointWorkspace:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("checkpoint workspace must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock.acquire()
        except Timeout as exc:
            raise ValueError(f"checkpoint workspace is already in use: {self.path}") from exc
        try:
            self._prepare()
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.lock.release()

    def _prepare(self) -> None:
        manifest = {
            "workspace_schema": WORKSPACE_SCHEMA,
            "execution_contract": EXECUTION_CONTRACT,
            "experiment_identity": self.identity,
            "dataset_identity": self.dataset.identity,
            "tasks": self.task_keys,
        }
        if self.resume:
            if not self.path.is_dir() or self.path.is_symlink():
                raise ValueError(f"resume workspace does not exist: {self.path}")
            current = _read_regular_json(self.path / "workspace.json")
            if current != manifest:
                raise ValueError(
                    "incompatible checkpoint workspace: experiment/configuration changed"
                )
        else:
            if self.path.exists() or self.path.is_symlink():
                raise ValueError(f"checkpoint workspace already exists; use --resume: {self.path}")
            self.path.mkdir()
            (self.path / "tasks").mkdir()
            _write_atomic(self.path / "workspace.json", manifest)
        task_dir = self.path / "tasks"
        if task_dir.is_symlink() or not task_dir.is_dir():
            raise ValueError("checkpoint task directory is unsafe or missing")
        expected = {f"{value}.json" for value in self.task_keys.values()}
        for entry in task_dir.iterdir():
            if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                # Unpublished temporary bytes after an interrupted atomic write.
                continue
            if entry.name not in expected or entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected checkpoint workspace member: {entry.name}")

    def _path_for(self, model: ModelConfig) -> Path:
        return self.path / "tasks" / f"{self.task_keys[model.id or model.name]}.json"

    def load(self, model: ModelConfig) -> CandidateResult | None:
        """Reuse only verified, fully compatible results; retrain invalid entries."""
        from benchforge.execution.benchmark import CandidateResult

        identifier = model.id or model.name
        path = self._path_for(model)
        if not path.exists() and not path.is_symlink():
            return None
        try:
            data = _read_regular_json(path)
            if not isinstance(data, dict) or set(data) != {
                "schema", "experiment", "task", "payload", "sha256"
            }:
                raise ValueError("invalid checkpoint envelope")
            if data["schema"] != WORKSPACE_SCHEMA:
                raise ValueError("unsupported checkpoint schema")
            if data["experiment"] != self.identity or data["task"] != self.task_keys[identifier]:
                raise ValueError("checkpoint identity mismatch")
            if _sha(data["payload"]) != data["sha256"]:
                raise ValueError("checkpoint digest mismatch")
            payload = data["payload"]
            if not isinstance(payload, dict):
                raise ValueError("invalid checkpoint payload")
            if payload["model_identifier"] != identifier:
                raise ValueError("model identifier mismatch")
            if ModelConfig.model_validate(payload["model_config"]) != model:
                raise ValueError("candidate configuration mismatch")
            run = _RUN_ADAPTER.validate_python(payload["run_result"])
            expected_run = self.config.to_run_config(model)
            if (
                run.fingerprint != expected_run.fingerprint_for_dataset(self.dataset.identity)
                or run.dataset_identity != self.dataset.identity
                or run.model_name != model.name
                or run.seed != self.config.seed
                or run.artifact_directory is not None
            ):
                raise ValueError("run semantic identity mismatch")
            if [f.fold_id for f in run.fold_results] != [f.fold_id for f in self.folds]:
                raise ValueError("fold identity mismatch")
            expected_indices = {
                int(idx): fold.fold_id
                for fold in self.folds
                for idx in fold.validation_indices
            }
            if len(run.predictions) != len(expected_indices) or any(
                expected_indices.get(record.sample_index) != record.fold_id
                for record in run.predictions
            ):
                raise ValueError("validation prediction/fold assignment mismatch")
            if len({record.sample_index for record in run.predictions}) != len(run.predictions):
                raise ValueError("duplicate prediction index")
            if aggregate_fold_metrics(f.metrics for f in run.fold_results) != run.aggregate_metrics:
                raise ValueError("aggregate scores disagree with saved fold scores")
            duration = float(payload["duration_seconds"])
            if not np.isfinite(duration) or duration < 0:
                raise ValueError("invalid candidate duration")
            candidate = CandidateResult(identifier, model, run, duration)
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
            self.invalid.append(f"{identifier}: {type(exc).__name__}: {exc}")
            return None
        self.reused.append(identifier)
        return candidate

    def save(self, candidate: CandidateResult) -> None:
        """Only successful, complete results may become reusable checkpoints."""
        identifier = candidate.model_identifier
        payload = {
            "model_identifier": identifier,
            "model_config": candidate.model_config.model_dump(mode="json"),
            "run_result": asdict(candidate.run_result),
            "duration_seconds": candidate.duration_seconds,
        }
        envelope = {
            "schema": WORKSPACE_SCHEMA,
            "experiment": self.identity,
            "task": self.task_keys[identifier],
            "payload": payload,
            "sha256": _sha(payload),
        }
        _write_atomic(self._path_for(candidate.model_config), envelope)
        self.executed.append(identifier)
