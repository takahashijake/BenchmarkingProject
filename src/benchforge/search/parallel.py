"""Immutable task planning and coordinator-owned parallel search orchestration."""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import dataclass

import numpy as np

from benchforge.core.config import SearchConfig
from benchforge.data.registry import Dataset
from benchforge.execution.runner import FoldExecutionError
from benchforge.execution.supervisor import ExecutionOptions, supervise
from benchforge.models.registry import default_model_registry
from benchforge.search.recovery import SearchEvidenceValidator, SearchWorkspace
from benchforge.search.registry import default_search_space_registry
from benchforge.search.results import (
    FamilySearchFailure,
    FamilySearchResult,
    OuterFoldSearchResult,
    SearchTaskDiagnostic,
    TrialRecord,
)
from benchforge.splits.stratified import Fold


def validate_workers(workers: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    available = os.cpu_count() or 1
    if workers > available:
        raise ValueError(f"requested {workers} workers but only {available} logical CPUs detected")


@dataclass(frozen=True)
class FoldAssignment:
    fold_id: int
    train: tuple[int, ...]
    validation: tuple[int, ...]

    @classmethod
    def from_fold(cls, fold: Fold) -> FoldAssignment:
        return cls(
            fold.fold_id,
            tuple(map(int, fold.train_indices)),
            tuple(map(int, fold.validation_indices)),
        )

    def to_fold(self) -> Fold:
        train = np.asarray(self.train, dtype=np.int64)
        validation = np.asarray(self.validation, dtype=np.int64)
        train.setflags(write=False)
        validation.setflags(write=False)
        return Fold(self.fold_id, train, validation)


@dataclass(frozen=True)
class OuterTask:
    task_id: str
    candidate_index: int
    outer: FoldAssignment
    inner: tuple[FoldAssignment, ...]


@dataclass(frozen=True)
class SearchExecutionPlan:
    tasks: tuple[OuterTask, ...]

    @classmethod
    def build(
        cls, config: SearchConfig, outer: tuple[Fold, ...], inner: dict[int, tuple[Fold, ...]]
    ) -> SearchExecutionPlan:
        return cls(
            tuple(
                OuterTask(
                    SearchWorkspace.outer_key(candidate, fold),
                    index,
                    FoldAssignment.from_fold(fold),
                    tuple(FoldAssignment.from_fold(f) for f in inner[fold.fold_id]),
                )
                for index, candidate in enumerate(config.models)
                for fold in outer
            )
        )


@dataclass(frozen=True)
class TaskRequest:
    config: SearchConfig
    dataset: Dataset
    task: OuterTask


@dataclass(frozen=True)
class TaskOutcome:
    result: OuterFoldSearchResult | None = None
    error_type: str | None = None
    message: str = ""
    trials: tuple[TrialRecord, ...] = ()
    configured_trials: int = 0
    unexpected: bool = False


def _outer_process_entry(request: TaskRequest) -> TaskOutcome:
    """Trusted worker boundary. No workspace path, locking or artifact writes."""
    from threadpoolctl import threadpool_limits

    from benchforge.search.runner import FamilySearchError, OuterRefitError, _search_outer

    try:
        with threadpool_limits(limits=1):
            result = _search_outer(
                request.config,
                request.config.models[request.task.candidate_index],
                request.dataset,
                request.task.outer.to_fold(),
                tuple(f.to_fold() for f in request.task.inner),
                model_registry=default_model_registry,
                search_space_registry=default_search_space_registry,
            )
        return TaskOutcome(result=result)
    except (ValueError, FoldExecutionError, FamilySearchError) as exc:
        return TaskOutcome(
            error_type=(
                "FoldExecutionError" if isinstance(exc, OuterRefitError) else type(exc).__name__
            ),
            message=str(exc),
            trials=exc.trials if isinstance(exc, (FamilySearchError, OuterRefitError)) else (),
            configured_trials=(
                exc.configured_trials
                if isinstance(exc, (FamilySearchError, OuterRefitError))
                else 0
            ),
        )
    except Exception as exc:
        # Carry unexpected Python defects to the coordinator; nonzero exits are crashes.
        return TaskOutcome(error_type=type(exc).__name__, message=str(exc), unexpected=True)


def search_parallel(
    config: SearchConfig,
    dataset: Dataset,
    outer: tuple[Fold, ...],
    inner: dict[int, tuple[Fold, ...]],
    *,
    workers: int,
    workspace: SearchWorkspace | None,
    diagnostics: list[SearchTaskDiagnostic],
) -> tuple[list[FamilySearchResult], list[FamilySearchFailure]]:
    from benchforge.search.runner import _aggregate_family, _counts

    plan = SearchExecutionPlan.build(config, outer, inner)
    evidence = workspace or SearchEvidenceValidator(
        config, dataset, inner, default_search_space_registry
    )
    outcomes: dict[str, TaskOutcome] = {}
    elapsed: dict[str, float] = {}
    pending: list[TaskRequest] = []
    for task in plan.tasks:
        candidate = config.models[task.candidate_index]
        value = workspace.load_outer(candidate, task.outer.to_fold()) if workspace else None
        if value is None:
            pending.append(TaskRequest(config, dataset, task))
        else:
            outcomes[task.task_id] = TaskOutcome(result=value)
            elapsed[task.task_id] = 0.0
            diagnostics.append(SearchTaskDiagnostic(task.task_id, "reused"))

    if pending:
        with closing(
            supervise(
                _outer_process_entry,
                pending,
                workers=workers,
                options=ExecutionOptions(),
                deadline=None,
            )
        ) as completions:
            for completed in completions:
                task = pending[completed.index].task
                if completed.reason is not None:
                    outcome = TaskOutcome(
                        error_type=completed.reason,
                        message=f"worker exited with status {completed.exit_code}",
                    )
                else:
                    received = completed.result
                    if not isinstance(received, TaskOutcome):
                        raise RuntimeError(f"invalid worker IPC for {task.task_id}")
                    outcome = received
                    if outcome.unexpected:
                        raise RuntimeError(
                            f"unexpected worker error for {task.task_id}: "
                            f"{outcome.error_type}: {outcome.message}"
                        )
                    if (outcome.result is None) == (outcome.error_type is None):
                        raise RuntimeError(f"inconsistent worker IPC for {task.task_id}")
                if outcome.result is not None:
                    candidate = config.models[task.candidate_index]
                    fold = task.outer.to_fold()
                    # Validate even when no workspace was requested. Validation errors are fatal.
                    if workspace is not None:
                        workspace.save_outer(candidate, fold, outcome.result)
                    else:
                        evidence.outer_payload(candidate, fold, outcome.result)
                outcomes[task.task_id] = outcome
                elapsed[task.task_id] = completed.elapsed
                diagnostics.append(
                    SearchTaskDiagnostic(
                        task.task_id, "success" if outcome.result is not None else "failed"
                    )
                )

    families: list[FamilySearchResult] = []
    failures: list[FamilySearchFailure] = []
    for index, candidate in enumerate(config.models):
        ordered = [outcomes[t.task_id] for t in plan.tasks if t.candidate_index == index]
        failed = [o for o in ordered if o.error_type is not None]
        folds = [o.result for o in ordered if o.result is not None]
        # Parallel family durations sum newly executed task wall times (including
        # failures/startup); the result's total duration is coordinator wall time.
        duration = sum(elapsed[t.task_id] for t in plan.tasks if t.candidate_index == index)
        if failed:
            trials = tuple(
                t for o in ordered for t in (o.result.trials if o.result is not None else o.trials)
            )
            completed_trials, failed_trials, pruned_trials = _counts(trials)
            failures.append(
                FamilySearchFailure(
                    model_identifier=candidate.id or candidate.name,
                    model_name=candidate.name,
                    error_type=failed[0].error_type or "WorkerCrash",
                    message=failed[0].message,
                    duration_seconds=duration,
                    configured_trials=sum(
                        o.result.configured_trials if o.result is not None else o.configured_trials
                        for o in ordered
                    ),
                    completed_trials=completed_trials,
                    failed_trials=failed_trials,
                    pruned_trials=pruned_trials,
                    trials=trials,
                )
            )
        else:
            families.append(_aggregate_family(candidate, folds, duration))
    # Expose diagnostic order independent of completion order too.
    position = {t.task_id: i for i, t in enumerate(plan.tasks)}
    diagnostics.sort(key=lambda d: position[d.task_id])
    return families, failures
