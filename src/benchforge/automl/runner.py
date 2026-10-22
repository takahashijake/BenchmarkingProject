from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from benchforge.automl.planner import build_automl_plan
from benchforge.automl.results import AutoMLResult
from benchforge.core.config import AutoMLConfig
from benchforge.data.registry import DatasetRegistry, default_dataset_registry
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.search.registry import SearchSpaceRegistry, default_search_space_registry
from benchforge.search.runner import _run_search


def _run_automl(
    config: AutoMLConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    persist: bool = True,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    _stack: ExitStack,
) -> AutoMLResult:
    """Plan AutoML deterministically, then delegate all evaluation to ``run_search``."""
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

    started = perf_counter()
    dataset = dataset_registry.resolve(config.dataset)
    plan = build_automl_plan(
        config,
        dataset_registry=dataset_registry,
        model_registry=model_registry,
        search_space_registry=search_space_registry,
        dataset=dataset,
    )
    search_result = _run_search(
        plan.search_config,
        dataset_registry=dataset_registry,
        model_registry=model_registry,
        search_space_registry=search_space_registry,
        persist=False,
        dataset=dataset,
        _stack=_stack,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        _orchestration={
            "automl_plan": plan.canonical_dict(),
            "selected_models": plan.selected_model_identity,
        },
    )
    result = AutoMLResult(
        fingerprint=plan.fingerprint,
        plan=plan,
        search_result=search_result,
        total_duration_seconds=perf_counter() - started,
    )
    if persist:
        from benchforge.storage.automl import AutoMLArtifactStore

        automl_directory, search_directory = AutoMLArtifactStore(config.output.directory).write(
            config, result
        )
        result = replace(
            result,
            search_result=replace(search_result, artifact_directory=search_directory),
            artifact_directory=automl_directory,
        )
    return result


def run_automl(
    config: AutoMLConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    persist: bool = True,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> AutoMLResult:
    """Plan deterministically and recover through ordinary search, including parent sealing."""
    with ExitStack() as stack:
        return _run_automl(
            config,
            dataset_registry=dataset_registry,
            model_registry=model_registry,
            search_space_registry=search_space_registry,
            persist=persist,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            _stack=stack,
        )
