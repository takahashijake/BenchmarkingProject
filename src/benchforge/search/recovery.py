"""Complete-study recovery. No sampler state or executable model state is persisted."""

from __future__ import annotations

import platform
from dataclasses import asdict, fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, TypeAdapter

from benchforge._version import __version__
from benchforge.core.config import JsonScalar, SearchConfig, SearchModelConfig
from benchforge.core.seeds import derive_seed
from benchforge.data.identity import dataset_digest
from benchforge.data.registry import Dataset, summarize_dataset
from benchforge.evaluation.metrics import compute_metrics
from benchforge.execution.checkpoint_io import _read_regular_json, _sha, _write_atomic
from benchforge.execution.runner import FoldResult, PredictionRecord
from benchforge.search.registry import SearchSpaceRegistry
from benchforge.search.results import FinalCandidate, OuterFoldSearchResult, TrialRecord
from benchforge.splits.stratified import Fold, build_folds

SCHEMA = 1
CONTRACT = "nested-search-complete-outer-fold-v1"


class CheckpointPublicationError(RuntimeError):
    """Publication failures abort orchestration, never become scientific family failures."""


def fold_plan(folds: tuple[Fold, ...]) -> list[dict[str, Any]]:
    return [
        dict(
            fold_id=f.fold_id,
            train=f.train_indices.tolist(),
            validation=f.validation_indices.tolist(),
        )
        for f in folds
    ]


class StoredOuterFold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fold_id: int
    outer_train_indices: tuple[int, ...]
    outer_validation_indices: tuple[int, ...]
    inner_plan: list[dict[str, Any]]
    estimator_seed: int
    selected_parameters: dict[str, JsonScalar]
    best_inner_score: float | None
    trials: tuple[TrialRecord, ...]
    outer_result: FoldResult
    outer_predictions: tuple[PredictionRecord, ...]
    duration_seconds: float
    configured_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    sampler_seed: int | None


_OUTER = TypeAdapter(StoredOuterFold)
_FINAL = TypeAdapter(FinalCandidate)


class SearchEvidenceValidator:
    """Shared scientific validation, independent of workspace I/O or locking."""

    def __init__(
        self,
        config: SearchConfig,
        dataset: Dataset,
        inner: dict[int, tuple[Fold, ...]],
        spaces: SearchSpaceRegistry,
    ) -> None:
        self.config, self.dataset, self.inner, self.spaces = config, dataset, inner, spaces

    def _trials(
        self,
        candidate: SearchModelConfig,
        records: tuple[TrialRecord, ...],
        configured: int,
        completed: int,
        failed: int,
        pruned: int,
        selected: dict[str, JsonScalar],
        score: float | None,
        budget: int,
    ) -> None:
        expected_budget = budget if candidate.mode == "search" else 0
        if (
            configured != expected_budget
            or len(records) != configured
            or [r.number for r in records] != list(range(configured))
            or (completed, failed, pruned)
            != tuple(
                sum(r.state == state for r in records) for state in ("COMPLETE", "FAIL", "PRUNED")
            )
        ):
            raise ValueError("invalid trial counts or sequence")
        if candidate.mode == "fixed":
            if _sha(selected) != _sha(candidate.parameters):
                raise ValueError("fixed parameters changed")
            return
        space = self.spaces.resolve(candidate.name)
        for r in records:
            if r.state not in {"COMPLETE", "FAIL"} or r.duration_seconds < 0:
                raise ValueError("invalid trial state or duration")
            if set(r.proposed_parameters) & set(candidate.parameters):
                raise ValueError("trial overwrites fixed parameters")
            if _sha(r.parameters) != _sha(
                {**space.defaults, **candidate.parameters, **r.proposed_parameters}
            ):
                raise ValueError("trial parameters disagree with model configuration")
            if (r.state == "COMPLETE") != (r.value is not None):
                raise ValueError("invalid trial objective")
        successes = [r for r in records if r.state == "COMPLETE"]
        if not successes:
            raise ValueError("checkpoint has no successful trial")
        direction = self.config.metric_spec(self.config.primary_metric).direction
        best = sorted(
            successes,
            key=lambda r: (
                -float(r.value if r.value is not None else 0)
                if direction == "maximize"
                else float(r.value if r.value is not None else 0),
                r.number,
            ),
        )[0]
        if _sha(selected) != _sha(best.parameters) or score != best.value:
            raise ValueError("selection disagrees with trial evidence")

    def _validate_outer(
        self, candidate: SearchModelConfig, fold: Fold, value: StoredOuterFold
    ) -> OuterFoldSearchResult:
        inner = self.inner[fold.fold_id] if candidate.mode == "search" else ()
        seed = (
            derive_seed(self.config.seed, "sampler", candidate.id or candidate.name, fold.fold_id)
            if candidate.mode == "search"
            else None
        )
        if (
            value.fold_id != fold.fold_id
            or value.outer_train_indices != tuple(fold.train_indices)
            or value.outer_validation_indices != tuple(fold.validation_indices)
            or _sha(value.inner_plan) != _sha(fold_plan(inner))
            or value.estimator_seed
            != derive_seed(
                self.config.seed, "estimator", candidate.id or candidate.name, fold.fold_id
            )
            or value.sampler_seed != seed
        ):
            raise ValueError("checkpoint fold plan or seed mismatch")
        self._trials(
            candidate,
            value.trials,
            value.configured_trials,
            value.completed_trials,
            value.failed_trials,
            value.pruned_trials,
            value.selected_parameters,
            value.best_inner_score,
            self.config.budget.trials_per_outer_fold,
        )
        if candidate.mode == "fixed" and value.best_inner_score is not None:
            raise ValueError("fixed outer task has inner selection evidence")
        result, predictions = value.outer_result, value.outer_predictions
        if (
            result.fold_id != fold.fold_id
            or result.train_size != len(fold.train_indices)
            or result.validation_size != len(fold.validation_indices)
            or len(predictions) != len(fold.validation_indices)
            or [p.sample_index for p in predictions] != fold.validation_indices.tolist()
            or any(p.fold_id != fold.fold_id for p in predictions)
        ):
            raise ValueError("invalid prediction indices or fold membership")
        truth = self.dataset.target.iloc[fold.validation_indices].to_numpy()
        if not np.array_equal(truth, [p.truth for p in predictions]):
            raise ValueError("prediction truth disagrees with dataset")
        predicted = np.asarray([p.prediction for p in predictions])
        if not np.isfinite(predicted).all():
            raise ValueError("nonfinite predictions")
        if self.config.task.is_classification and not set(predicted) <= set(self.dataset.target):
            raise ValueError("invalid predicted class")
        needs_score = any(self.config.metric_spec(m).requires_score for m in self.config.metrics)
        if any((p.score is not None) != needs_score for p in predictions):
            raise ValueError("invalid prediction score availability")
        scores = np.asarray([p.score for p in predictions], dtype=float) if needs_score else None
        if scores is not None and not np.isfinite(scores).all():
            raise ValueError("nonfinite prediction score")
        metrics = compute_metrics(
            self.config.metrics,
            truth,
            predicted,
            scores,
            self.config.task,
            manifests=self.config.provider_manifests,
            functions=(
                self.config._provider_runtime.metrics if self.config._provider_runtime else None
            ),
        )
        if set(result.metrics) != set(metrics) or any(
            not np.isclose(result.metrics[k], v, rtol=1e-12, atol=1e-12) for k, v in metrics.items()
        ):
            raise ValueError("checkpoint metrics disagree with predictions")
        if value.duration_seconds < 0:
            raise ValueError("invalid duration")
        values = {
            name: getattr(value, name)
            for name in type(value).model_fields
            if name not in {"inner_plan", "estimator_seed"}
        }
        return OuterFoldSearchResult(**values, inner_folds=inner)

    def outer_payload(
        self, candidate: SearchModelConfig, fold: Fold, value: OuterFoldSearchResult
    ) -> StoredOuterFold:
        payload = {f.name: getattr(value, f.name) for f in fields(value) if f.name != "inner_folds"}
        payload["inner_plan"] = fold_plan(value.inner_folds)
        payload["estimator_seed"] = derive_seed(
            self.config.seed, "estimator", candidate.id or candidate.name, fold.fold_id
        )
        wire = _OUTER.validate_python(payload)
        self._validate_outer(candidate, fold, wire)
        return wire


class SearchWorkspace(SearchEvidenceValidator):
    """Exclusive coordinator-owned JSON workspace, independent of published artifacts."""

    def __init__(
        self,
        path: Path,
        config: SearchConfig,
        dataset: Dataset,
        outer: tuple[Fold, ...],
        inner: dict[int, tuple[Fold, ...]],
        spaces: SearchSpaceRegistry,
        model_identity: dict[str, Any],
        *,
        resume: bool,
        orchestration: dict[str, Any] | None = None,
    ) -> None:
        self.path = path.absolute()
        super().__init__(config, dataset, inner, spaces)
        self.outer = outer
        self.resume = resume
        self.reused: list[str] = []
        self.executed: list[str] = []
        self.invalid: list[str] = []
        searchable = {m.name for m in config.models if m.mode == "search"}
        self.final_plans = (
            {
                self.final_key(m): {
                    "folds": fold_plan(
                        build_folds(
                            dataset.target,
                            config.inner_split,
                            derive_seed(config.seed, "final", "folds", m.id or m.name),
                            config.task,
                        )
                    ),
                    "estimator_seed": derive_seed(
                        config.seed, "final", "estimator", m.id or m.name
                    ),
                    "sampler_seed": derive_seed(config.seed, "final", "sampler", m.id or m.name),
                }
                for m in config.models
            }
            if config.final_search.enabled
            else {}
        )
        self.identity = _sha(
            {
                "schema": SCHEMA,
                "contract": CONTRACT,
                "search": config.fingerprint_for_dataset(
                    dataset.identity, spaces.identity_for(searchable)
                ),
                "dataset_content": dataset_digest(dataset),
                "dataset_summary": asdict(summarize_dataset(dataset)),
                "outer": fold_plan(outer),
                "inner": {str(k): fold_plan(v) for k, v in inner.items()},
                "models": model_identity,
                "final_plans": self.final_plans,
                "space_defaults": {n: spaces.resolve(n).defaults for n in sorted(searchable)},
                "orchestration": orchestration,
                "versions": {
                    name: version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "scipy", "optuna")
                },
                "python": platform.python_version(),
                "benchforge": __version__,
            }
        )
        names = [self.outer_key(m, f) for m in config.models for f in outer]
        if config.final_search.enabled:
            names += [self.final_key(m) for m in config.models]
        self.tasks = {name: _sha({"experiment": self.identity, "unit": name}) for name in names}
        self.lock = FileLock(str(self.path) + ".lock", timeout=0)

    @staticmethod
    def outer_key(candidate: SearchModelConfig, fold: Fold) -> str:
        return f"outer:{candidate.id or candidate.name}:{fold.fold_id}"

    @staticmethod
    def final_key(candidate: SearchModelConfig) -> str:
        return f"final:{candidate.id or candidate.name}"

    def __enter__(self) -> SearchWorkspace:
        # Check every ancestor, including the lock path, before opening anything.
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError("checkpoint workspace must not contain symlink ancestors")
        lock_path = Path(str(self.path) + ".lock")
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise ValueError("unsafe checkpoint lock file")
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
        manifest = dict(
            workspace_schema=SCHEMA,
            execution_contract=CONTRACT,
            experiment_identity=self.identity,
            tasks=self.tasks,
        )
        if self.resume:
            if not self.path.is_dir():
                raise ValueError("resume workspace does not exist")
            if _sha(_read_regular_json(self.path / "workspace.json")) != _sha(manifest):
                raise ValueError("incompatible search workspace: scientific experiment changed")
        else:
            if self.path.exists():
                raise ValueError("checkpoint workspace already exists; use --resume")
            self.path.mkdir()
            (self.path / "tasks").mkdir()
            _write_atomic(self.path / "workspace.json", manifest)
        for entry in self.path.iterdir():
            if entry.is_symlink() or entry.name not in {"workspace.json", "tasks"}:
                raise ValueError(f"unexpected checkpoint workspace member: {entry.name}")
        tasks = self.path / "tasks"
        if not tasks.is_dir() or tasks.is_symlink():
            raise ValueError("unsafe checkpoint tasks directory")
        expected = {f"{key}.json" for key in self.tasks.values()}
        for entry in tasks.iterdir():
            if (
                entry.is_symlink()
                or not entry.is_file()
                or (
                    entry.name not in expected
                    and not (entry.name.startswith(".") and entry.name.endswith(".tmp"))
                )
            ):
                raise ValueError(f"unexpected checkpoint workspace member: {entry.name}")

    def _path(self, key: str) -> Path:
        return self.path / "tasks" / f"{self.tasks[key]}.json"

    def _load(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            return None
        data = _read_regular_json(path)
        if not isinstance(data, dict) or set(data) != {
            "schema",
            "contract",
            "experiment",
            "task",
            "payload",
            "sha256",
        }:
            raise ValueError("invalid checkpoint envelope")
        if (
            type(data["schema"]) is not int
            or data["schema"] != SCHEMA
            or data["contract"] != CONTRACT
            or data["experiment"] != self.identity
            or data["task"] != self.tasks[key]
            or data["sha256"] != _sha(data["payload"])
        ):
            raise ValueError("checkpoint version, identity or digest mismatch")
        if not isinstance(data["payload"], dict):
            raise ValueError("checkpoint payload must be an object")
        return data["payload"]

    def _save(self, key: str, payload: Any) -> None:
        _write_atomic(
            self._path(key),
            dict(
                schema=SCHEMA,
                contract=CONTRACT,
                experiment=self.identity,
                task=self.tasks[key],
                payload=payload,
                sha256=_sha(payload),
            ),
        )
        self.executed.append(key)

    def load_outer(self, candidate: SearchModelConfig, fold: Fold) -> OuterFoldSearchResult | None:
        key = self.outer_key(candidate, fold)
        try:
            payload = self._load(key)
            if payload is None:
                return None
            wire = _OUTER.validate_python(payload)
            if _sha(_OUTER.dump_python(wire, mode="json")) != _sha(payload):
                raise ValueError("noncanonical or unexpected checkpoint fields")
            value = self._validate_outer(candidate, fold, wire)
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid.append(f"{key}: {exc}")
            return None
        self.reused.append(key)
        return value

    def save_outer(
        self, candidate: SearchModelConfig, fold: Fold, value: OuterFoldSearchResult
    ) -> None:
        # The same validation is mandatory at publication and reuse.
        try:
            wire = self.outer_payload(candidate, fold, value)
            self._save(self.outer_key(candidate, fold), _OUTER.dump_python(wire, mode="json"))
        except (ValueError, TypeError) as exc:
            raise CheckpointPublicationError("outer-fold checkpoint publication failed") from exc

    def _validate_final(self, candidate: SearchModelConfig, value: FinalCandidate) -> None:
        if (
            value.model_identifier != (candidate.id or candidate.name)
            or value.model_name != candidate.name
            or value.mode != candidate.mode
            or value.duration_seconds < 0
            or not np.isfinite(value.cv_selection_score)
        ):
            raise ValueError("final candidate identity mismatch")
        self._trials(
            candidate,
            value.trials,
            value.configured_trials,
            value.completed_trials,
            value.failed_trials,
            value.pruned_trials,
            value.parameters,
            value.cv_selection_score,
            self.config.final_search.trials,
        )

    def load_final(self, candidate: SearchModelConfig) -> FinalCandidate | None:
        key = self.final_key(candidate)
        try:
            payload = self._load(key)
            if payload is None:
                return None
            if set(payload) != {"result", "plan"} or _sha(payload["plan"]) != _sha(
                self.final_plans[key]
            ):
                raise ValueError("final checkpoint plan mismatch")
            value = _FINAL.validate_python(payload["result"])
            if _sha(_FINAL.dump_python(value, mode="json")) != _sha(payload["result"]):
                raise ValueError("noncanonical final checkpoint fields")
            self._validate_final(candidate, value)
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid.append(f"{key}: {exc}")
            return None
        self.reused.append(key)
        return value

    def save_final(self, candidate: SearchModelConfig, value: FinalCandidate) -> None:
        try:
            self._validate_final(candidate, value)
            key = self.final_key(candidate)
            self._save(
                key,
                {"result": _FINAL.dump_python(value, mode="json"), "plan": self.final_plans[key]},
            )
        except (ValueError, TypeError) as exc:
            raise CheckpointPublicationError("final checkpoint publication failed") from exc
