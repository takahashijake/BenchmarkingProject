from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import optuna

from benchforge.core.config import (
    JsonScalar,
    ModelConfig,
    SearchConfig,
    SearchModelConfig,
    TaskType,
)
from benchforge.core.seeds import derive_seed as derive_seed
from benchforge.data.registry import (
    Dataset,
    DatasetRegistry,
    default_dataset_registry,
    summarize_dataset,
)
from benchforge.evaluation.metrics import aggregate_fold_metrics
from benchforge.execution.runner import FoldExecutionError, execute_fold, run_benchmark
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.search.recovery import SearchWorkspace
from benchforge.search.registry import SearchSpaceRegistry, default_search_space_registry
from benchforge.search.results import (
    FamilySearchFailure,
    FamilySearchResult,
    FinalCandidate,
    OuterFoldSearchResult,
    SearchResult,
    SearchTaskDiagnostic,
    TrialRecord,
    TunedLeaderboardEntry,
)
from benchforge.search.spaces import SearchSpace
from benchforge.splits.stratified import Fold, build_folds, build_inner_folds


class TrialEvaluationError(RuntimeError):
    """A recoverable estimator/trial failure recorded by Optuna."""


class FamilySearchError(RuntimeError):
    """A family cannot produce a selected configuration."""

    def __init__(
        self,
        message: str,
        *,
        trials: tuple[TrialRecord, ...] = (),
        configured_trials: int = 0,
    ) -> None:
        super().__init__(message)
        self.trials = trials
        self.configured_trials = configured_trials


class OuterRefitError(FoldExecutionError):
    """Retain the completed study when its selected configuration cannot refit."""

    def __init__(self, message: str, trials: tuple[TrialRecord, ...], configured: int) -> None:
        super().__init__(message)
        self.trials = trials
        self.configured_trials = configured


def _parameters(
    space: SearchSpace,
    candidate: SearchModelConfig,
    proposed: dict[str, JsonScalar],
) -> dict[str, JsonScalar]:
    return {**space.defaults, **candidate.parameters, **proposed}


def _validate_model_parameters(
    candidate: SearchModelConfig,
    parameters: dict[str, JsonScalar],
    seed: int,
    model_registry: ModelRegistry,
    task: TaskType,
) -> None:
    config = ModelConfig(name=candidate.name, id=candidate.id, parameters=parameters)
    model_registry.create(config, seed, task)
    model_registry.capabilities(candidate.name)


def _trial_records(study: optuna.study.Study) -> tuple[TrialRecord, ...]:
    records: list[TrialRecord] = []
    for trial in study.trials:
        duration = 0.0 if trial.duration is None else trial.duration.total_seconds()
        raw_parameters = trial.user_attrs.get("parameters", trial.params)
        parameters = {str(key): value for key, value in dict(raw_parameters).items()}
        proposed = {str(key): value for key, value in trial.params.items()}
        records.append(
            TrialRecord(
                number=trial.number,
                state=trial.state.name,
                proposed_parameters=proposed,
                parameters=parameters,
                value=None if trial.value is None else float(trial.value),
                duration_seconds=max(0.0, duration),
                error_type=trial.user_attrs.get("error_type"),
                error_message=trial.user_attrs.get("error_message"),
            )
        )
    return tuple(records)


def _run_study(
    config: SearchConfig,
    candidate: SearchModelConfig,
    dataset: Dataset,
    folds: tuple[Fold, ...],
    *,
    trials: int,
    timeout: float | None,
    sampler_seed: int,
    estimator_seed: int,
    space: SearchSpace,
    model_registry: ModelRegistry,
) -> tuple[dict[str, JsonScalar], float, tuple[TrialRecord, ...]]:
    base_parameters = _parameters(space, candidate, {})
    _validate_model_parameters(
        candidate, base_parameters, estimator_seed, model_registry, config.task
    )
    direction = config.metric_spec(config.primary_metric).direction

    def objective(trial: optuna.trial.Trial) -> float:
        proposed = space.propose(trial, dict(candidate.parameters))
        parameters = _parameters(space, candidate, proposed)
        trial.set_user_attr("parameters", parameters)
        model = ModelConfig(name=candidate.name, id=candidate.id, parameters=parameters)
        run_config = config.to_run_config(model, split=config.inner_split, seed=estimator_seed)
        try:
            result = run_benchmark(
                run_config,
                model_registry=model_registry,
                persist=False,
                dataset=dataset,
                folds=folds,
            )
        except FoldExecutionError as exc:
            root = exc.__cause__ or exc
            if not isinstance(root, (ValueError, ArithmeticError)):
                raise
            trial.set_user_attr("error_type", type(root).__name__)
            trial.set_user_attr("error_message", str(root))
            raise TrialEvaluationError(str(exc)) from exc
        return result.aggregate_metrics[config.primary_metric.value].mean

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        )
        study.optimize(
            objective,
            n_trials=trials,
            timeout=timeout,
            n_jobs=1,
            catch=(TrialEvaluationError,),
            show_progress_bar=False,
        )
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
    records = _trial_records(study)
    completed = [record for record in records if record.state == "COMPLETE"]
    if not completed:
        raise FamilySearchError(
            f"all {len(records)} trials failed or were pruned for '{candidate.name}'",
            trials=records,
            configured_trials=trials,
        )
    best = study.best_trial
    parameters = {str(key): value for key, value in best.user_attrs["parameters"].items()}
    if best.value is None:  # pragma: no cover - COMPLETE Optuna trials have a value
        raise FamilySearchError(f"best trial for '{candidate.name}' has no objective value")
    return parameters, float(best.value), records


def _counts(records: tuple[TrialRecord, ...]) -> tuple[int, int, int]:
    return (
        sum(record.state == "COMPLETE" for record in records),
        sum(record.state == "FAIL" for record in records),
        sum(record.state == "PRUNED" for record in records),
    )


def _search_outer(
    config: SearchConfig,
    candidate: SearchModelConfig,
    dataset: Dataset,
    outer_fold: Fold,
    inner_folds: tuple[Fold, ...],
    *,
    model_registry: ModelRegistry,
    search_space_registry: SearchSpaceRegistry,
) -> OuterFoldSearchResult:
    """The single scientific task kernel, shared by direct and isolated execution."""
    identifier = candidate.id or candidate.name
    space = search_space_registry.resolve(candidate.name) if candidate.mode == "search" else None
    if candidate.mode == "fixed":
        _validate_model_parameters(
            candidate, dict(candidate.parameters), config.seed, model_registry, config.task
        )
    fold_started = perf_counter()
    estimator_seed = derive_seed(config.seed, "estimator", identifier, outer_fold.fold_id)
    if space is not None:
        sampler_seed = derive_seed(config.seed, "sampler", identifier, outer_fold.fold_id)
        selected, inner_score, trials = _run_study(
            config,
            candidate,
            dataset,
            inner_folds,
            trials=config.budget.trials_per_outer_fold,
            timeout=config.budget.timeout_seconds_per_outer_fold,
            sampler_seed=sampler_seed,
            estimator_seed=estimator_seed,
            space=space,
            model_registry=model_registry,
        )
        configured_trials = config.budget.trials_per_outer_fold
    else:
        sampler_seed = None
        selected = dict(candidate.parameters)
        inner_score = None
        trials = ()
        configured_trials = 0

    model = ModelConfig(name=candidate.name, id=candidate.id, parameters=selected)
    run_config = config.to_run_config(model, split=config.outer_split, seed=estimator_seed)
    try:
        outer_result, predictions = execute_fold(
            run_config, dataset, outer_fold, model_registry=model_registry
        )
    except FoldExecutionError as exc:
        raise OuterRefitError(str(exc), trials, configured_trials) from exc
    completed, failed, pruned = _counts(trials)
    return OuterFoldSearchResult(
        fold_id=outer_fold.fold_id,
        outer_train_indices=tuple(int(index) for index in outer_fold.train_indices),
        outer_validation_indices=tuple(int(index) for index in outer_fold.validation_indices),
        inner_folds=inner_folds if space is not None else (),
        selected_parameters=selected,
        best_inner_score=inner_score,
        trials=trials,
        outer_result=outer_result,
        outer_predictions=predictions,
        duration_seconds=perf_counter() - fold_started,
        configured_trials=configured_trials,
        completed_trials=completed,
        failed_trials=failed,
        pruned_trials=pruned,
        sampler_seed=sampler_seed,
    )


def _aggregate_family(
    candidate: SearchModelConfig,
    fold_results: list[OuterFoldSearchResult],
    duration: float,
) -> FamilySearchResult:
    all_predictions = tuple(
        sorted(
            (prediction for fold in fold_results for prediction in fold.outer_predictions),
            key=lambda prediction: prediction.sample_index,
        )
    )
    configured = sum(fold.configured_trials for fold in fold_results)
    completed = sum(fold.completed_trials for fold in fold_results)
    failed = sum(fold.failed_trials for fold in fold_results)
    pruned = sum(fold.pruned_trials for fold in fold_results)
    return FamilySearchResult(
        model_identifier=candidate.id or candidate.name,
        candidate=candidate,
        outer_folds=tuple(fold_results),
        aggregate_metrics=aggregate_fold_metrics(
            fold.outer_result.metrics for fold in fold_results
        ),
        predictions=all_predictions,
        duration_seconds=duration,
        configured_trials=configured,
        completed_trials=completed,
        failed_trials=failed,
        pruned_trials=pruned,
    )


def _search_family(
    config: SearchConfig,
    candidate: SearchModelConfig,
    dataset: Dataset,
    outer_folds: tuple[Fold, ...],
    inner_folds: dict[int, tuple[Fold, ...]],
    *,
    model_registry: ModelRegistry,
    search_space_registry: SearchSpaceRegistry,
    workspace: SearchWorkspace | None = None,
    diagnostics: list[SearchTaskDiagnostic] | None = None,
) -> FamilySearchResult:
    started = perf_counter()
    fold_results: list[OuterFoldSearchResult] = []
    for outer_fold in outer_folds:
        key = SearchWorkspace.outer_key(candidate, outer_fold)
        recovered = workspace.load_outer(candidate, outer_fold) if workspace else None
        if recovered is not None:
            fold_results.append(recovered)
            if diagnostics is not None:
                diagnostics.append(SearchTaskDiagnostic(key, "reused"))
            continue
        try:
            result = _search_outer(
                config,
                candidate,
                dataset,
                outer_fold,
                inner_folds[outer_fold.fold_id],
                model_registry=model_registry,
                search_space_registry=search_space_registry,
            )
        except (ValueError, FoldExecutionError, FamilySearchError) as exc:
            if diagnostics is not None:
                diagnostics.append(SearchTaskDiagnostic(key, "failed"))
            if isinstance(exc, (FamilySearchError, OuterRefitError)):
                previous_trials = tuple(t for fold in fold_results for t in fold.trials)
                if isinstance(exc, OuterRefitError):
                    raise OuterRefitError(
                        str(exc),
                        (*previous_trials, *exc.trials),
                        sum(f.configured_trials for f in fold_results) + exc.configured_trials,
                    ) from exc
                raise FamilySearchError(
                    str(exc),
                    trials=(*previous_trials, *exc.trials),
                    configured_trials=(
                        sum(f.configured_trials for f in fold_results) + exc.configured_trials
                    ),
                ) from exc
            raise
        if workspace is not None:
            workspace.save_outer(candidate, outer_fold, result)
        fold_results.append(result)
        if diagnostics is not None:
            diagnostics.append(SearchTaskDiagnostic(key, "success"))
    return _aggregate_family(candidate, fold_results, perf_counter() - started)


def _leaderboard(
    families: list[FamilySearchResult],
    failures: list[FamilySearchFailure],
    config: SearchConfig,
) -> tuple[TunedLeaderboardEntry, ...]:
    name = config.primary_metric.value
    reverse = config.metric_spec(config.primary_metric).direction == "maximize"
    ordered = sorted(
        families,
        key=lambda family: (
            -family.aggregate_metrics[name].mean
            if reverse
            else family.aggregate_metrics[name].mean,
            family.model_identifier,
        ),
    )
    successful = [
        TunedLeaderboardEntry(
            rank=rank,
            model_identifier=family.model_identifier,
            mode=family.candidate.mode,
            status="success",
            primary_metric_mean=family.aggregate_metrics[name].mean,
            primary_metric_std=family.aggregate_metrics[name].std,
            metrics=family.aggregate_metrics,
            configured_trials=family.configured_trials,
            completed_trials=family.completed_trials,
            failed_trials=family.failed_trials,
            pruned_trials=family.pruned_trials,
            duration_seconds=family.duration_seconds,
        )
        for rank, family in enumerate(ordered, start=1)
    ]
    failed = [
        TunedLeaderboardEntry(
            rank=None,
            model_identifier=failure.model_identifier,
            mode=next(
                model.mode
                for model in config.models
                if (model.id or model.name) == failure.model_identifier
            ),
            status="failed",
            primary_metric_mean=None,
            primary_metric_std=None,
            metrics={},
            configured_trials=failure.configured_trials,
            completed_trials=failure.completed_trials,
            failed_trials=failure.failed_trials,
            pruned_trials=failure.pruned_trials,
            duration_seconds=failure.duration_seconds,
        )
        for failure in failures
    ]
    return tuple([*successful, *failed])


def _final_search(
    config: SearchConfig,
    family: FamilySearchResult,
    dataset: Dataset,
    *,
    model_registry: ModelRegistry,
    search_space_registry: SearchSpaceRegistry,
) -> FinalCandidate:
    started = perf_counter()
    candidate = family.candidate
    identifier = family.model_identifier
    folds = build_folds(
        dataset.target,
        config.inner_split,
        derive_seed(config.seed, "final", "folds", identifier),
        config.task,
    )
    estimator_seed = derive_seed(config.seed, "final", "estimator", identifier)
    if candidate.mode == "search":
        space = search_space_registry.resolve(candidate.name)
        selected, score, trials = _run_study(
            config,
            candidate,
            dataset,
            folds,
            trials=config.final_search.trials,
            timeout=config.final_search.timeout_seconds,
            sampler_seed=derive_seed(config.seed, "final", "sampler", identifier),
            estimator_seed=estimator_seed,
            space=space,
            model_registry=model_registry,
        )
        configured = config.final_search.trials
    else:
        selected = dict(candidate.parameters)
        trials = ()
        configured = 0
        model = ModelConfig(name=candidate.name, id=candidate.id, parameters=selected)
        result = run_benchmark(
            config.to_run_config(model, split=config.inner_split, seed=estimator_seed),
            model_registry=model_registry,
            persist=False,
            dataset=dataset,
            folds=folds,
        )
        score = result.aggregate_metrics[config.primary_metric.value].mean
    completed, failed, pruned = _counts(trials)
    return FinalCandidate(
        model_identifier=identifier,
        model_name=candidate.name,
        mode=candidate.mode,
        parameters=selected,
        cv_selection_score=score,
        trials=trials,
        configured_trials=configured,
        completed_trials=completed,
        failed_trials=failed,
        pruned_trials=pruned,
        duration_seconds=perf_counter() - started,
    )


def _run_search(
    config: SearchConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    persist: bool = True,
    dataset: Dataset | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    workers: int = 1,
    _orchestration: dict[str, Any] | None = None,
    _stack: ExitStack,
) -> SearchResult:
    """Tune families inside outer training partitions and rank outer-fold evidence only."""
    from benchforge.search.parallel import validate_workers

    validate_workers(workers)
    if workers > 1:
        if config.providers:
            raise ValueError("external providers require serial search execution")
        if (
            model_registry is not default_model_registry
            or search_space_registry is not default_search_space_registry
            or dataset_registry is not default_dataset_registry
        ):
            raise ValueError("parallel search requires built-in registries")
        if (
            config.budget.timeout_seconds_per_outer_fold is not None
            or config.final_search.timeout_seconds is not None
        ):
            raise ValueError("parallel search requires trial budgets without wall-clock timeouts")
        for candidate in config.models:
            jobs = candidate.parameters.get("n_jobs")
            if jobs is not None and jobs != 1:
                raise ValueError(f"model '{candidate.id or candidate.name}' requires n_jobs=1")

    from benchforge.providers.runtime import activate

    config, provider_runtime = activate(config)
    if provider_runtime is not None:
        if dataset_registry not in (default_dataset_registry, provider_runtime.datasets):
            raise ValueError("cannot combine provider activation with custom dataset registries")
        if model_registry not in (default_model_registry, provider_runtime.models):
            raise ValueError("cannot combine provider activation with custom model registries")
        dataset_registry = provider_runtime.datasets
        model_registry = provider_runtime.models
        if search_space_registry not in (default_search_space_registry, provider_runtime.spaces):
            raise ValueError("cannot combine providers with custom search-space registries")
        search_space_registry = provider_runtime.spaces

    if resume and checkpoint_dir is None:
        raise ValueError("resume requires checkpoint_dir")
    if checkpoint_dir is not None:
        if (
            config.budget.timeout_seconds_per_outer_fold is not None
            or config.final_search.timeout_seconds is not None
        ):
            raise ValueError("search recovery requires trial budgets without wall-clock timeouts")
        allowed_models = provider_runtime.models if provider_runtime else default_model_registry
        allowed_spaces = (
            provider_runtime.spaces if provider_runtime else default_search_space_registry
        )
        allowed_data = provider_runtime.datasets if provider_runtime else default_dataset_registry
        if (
            model_registry is not allowed_models
            or search_space_registry is not allowed_spaces
            or dataset_registry is not allowed_data
        ):
            raise ValueError("recovery requires built-in registries or version-pinned providers")

    started = perf_counter()
    dataset = dataset or dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError(
            f"dataset task '{dataset.task}' does not match configured task '{config.task}'"
        )
    searchable = {model.name for model in config.models if model.mode == "search"}
    space_identity = search_space_registry.identity_for(searchable)
    fingerprint = config.fingerprint_for_dataset(dataset.identity, space_identity)
    outer_folds = build_folds(dataset.target, config.outer_split, config.seed, config.task)
    inner_folds = {
        outer.fold_id: build_inner_folds(
            dataset.target,
            outer,
            config.inner_split,
            derive_seed(config.seed, "inner_folds", outer.fold_id),
            config.task,
        )
        for outer in outer_folds
    }
    workspace = None
    if checkpoint_dir is not None:
        workspace = _stack.enter_context(
            SearchWorkspace(
                checkpoint_dir,
                config,
                dataset,
                outer_folds,
                inner_folds,
                search_space_registry,
                model_registry.identity_for({m.name for m in config.models}),
                resume=resume,
                orchestration=_orchestration,
            )
        )
    diagnostics: list[SearchTaskDiagnostic] = []
    families: list[FamilySearchResult]
    failures: list[FamilySearchFailure]
    if workers > 1:
        from benchforge.search.parallel import search_parallel

        families, failures = search_parallel(
            config,
            dataset,
            outer_folds,
            inner_folds,
            workers=workers,
            workspace=workspace,
            diagnostics=diagnostics,
        )
    else:
        families = []
        failures = []
        for candidate in config.models:
            family_started = perf_counter()
            try:
                families.append(
                    _search_family(
                        config,
                        candidate,
                        dataset,
                        outer_folds,
                        inner_folds,
                        model_registry=model_registry,
                        search_space_registry=search_space_registry,
                        workspace=workspace,
                        diagnostics=diagnostics,
                    )
                )
            except (ValueError, FoldExecutionError, FamilySearchError) as exc:
                failure_trials = (
                    exc.trials if isinstance(exc, (FamilySearchError, OuterRefitError)) else ()
                )
                completed, failed, pruned = _counts(failure_trials)
                failures.append(
                    FamilySearchFailure(
                        model_identifier=candidate.id or candidate.name,
                        model_name=candidate.name,
                        error_type=(
                            "FoldExecutionError"
                            if isinstance(exc, OuterRefitError)
                            else type(exc).__name__
                        ),
                        message=str(exc),
                        duration_seconds=perf_counter() - family_started,
                        configured_trials=(
                            exc.configured_trials
                            if isinstance(exc, (FamilySearchError, OuterRefitError))
                            else 0
                        ),
                        completed_trials=completed,
                        failed_trials=failed,
                        pruned_trials=pruned,
                        trials=failure_trials,
                    )
                )
    leaderboard = _leaderboard(families, failures, config)
    successful = [entry for entry in leaderboard if entry.status == "success"]
    if not successful:
        details = "; ".join(
            f"{failure.model_identifier}: {failure.message}" for failure in failures
        )
        raise FamilySearchError(f"every search family failed: {details}")
    top_identifier = successful[0].model_identifier
    top_family = next(family for family in families if family.model_identifier == top_identifier)
    final_candidate = None
    if config.final_search.enabled:
        final_candidate = workspace.load_final(top_family.candidate) if workspace else None
        final_reused = final_candidate is not None
        if final_candidate is None:
            final_candidate = _final_search(
                config,
                top_family,
                dataset,
                model_registry=model_registry,
                search_space_registry=search_space_registry,
            )
            if workspace is not None:
                workspace.save_final(top_family.candidate, final_candidate)
        diagnostics.append(
            SearchTaskDiagnostic(
                SearchWorkspace.final_key(top_family.candidate),
                "reused" if final_reused else "success",
            )
        )
    result = SearchResult(
        provider_manifests=config.provider_manifests,
        fingerprint=fingerprint,
        search_space_identity=space_identity,
        dataset=summarize_dataset(dataset),
        primary_metric=config.primary_metric.value,
        optimization_direction=config.metric_spec(config.primary_metric).direction,
        outer_fold_count=len(outer_folds),
        inner_fold_count=config.inner_split.n_splits,
        outer_folds=outer_folds,
        families=tuple(families),
        failures=tuple(failures),
        leaderboard=leaderboard,
        final_candidate=final_candidate,
        total_duration_seconds=perf_counter() - started,
        checkpoint_reused=tuple(workspace.reused) if workspace else (),
        checkpoint_executed=tuple(workspace.executed) if workspace else (),
        checkpoint_invalid=tuple(workspace.invalid) if workspace else (),
        execution_mode="subprocess" if workers > 1 else "serial",
        execution_workers=workers,
        execution_planned_tasks=len(config.models) * len(outer_folds)
        + int(config.final_search.enabled),
        execution_diagnostics=tuple(diagnostics),
    )
    if persist:
        from benchforge.storage.search import SearchArtifactStore

        directory = SearchArtifactStore(config.output.directory).write(config, result)
        result = replace(result, artifact_directory=directory)
    return result


def run_search(
    config: SearchConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    persist: bool = True,
    dataset: Dataset | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    workers: int = 1,
) -> SearchResult:
    """Run nested search; optionally create or resume a complete-fold workspace.

    An interrupted study is recomputed. Recovery rejects wall-clock study limits
    and unversioned custom registries. The workspace lock spans artifact sealing.
    ``workers=1`` executes directly; larger counts isolate complete outer folds
    in bounded subprocesses and require built-ins and trial-count budgets.
    """
    with ExitStack() as stack:
        return _run_search(
            config,
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            search_space_registry=search_space_registry,
            persist=persist,
            dataset=dataset,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            workers=workers,
            _stack=stack,
        )
