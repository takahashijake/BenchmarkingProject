from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from benchforge.automl.planner import build_automl_plan
from benchforge.automl.results import AutoMLResult
from benchforge.core.config import AutoMLConfig
from benchforge.data.registry import DatasetRegistry, default_dataset_registry
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.search.registry import SearchSpaceRegistry, default_search_space_registry
from benchforge.search.runner import run_search


def run_automl(
    config: AutoMLConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    persist: bool = True,
) -> AutoMLResult:
    """Plan AutoML deterministically, then delegate all evaluation to ``run_search``."""
    started = perf_counter()
    dataset = dataset_registry.resolve(config.dataset)
    plan = build_automl_plan(
        config,
        dataset_registry=dataset_registry,
        model_registry=model_registry,
        search_space_registry=search_space_registry,
        dataset=dataset,
    )
    search_result = run_search(
        plan.search_config,
        dataset_registry=dataset_registry,
        model_registry=model_registry,
        search_space_registry=search_space_registry,
        persist=False,
        dataset=dataset,
    )
    result = AutoMLResult(
        fingerprint=plan.fingerprint,
        plan=plan,
        search_result=search_result,
        total_duration_seconds=perf_counter() - started,
    )
    if persist:
        from benchforge.storage.automl import AutoMLArtifactStore

        automl_directory, search_directory = AutoMLArtifactStore(
            config.output.directory
        ).write(config, result)
        result = replace(
            result,
            search_result=replace(search_result, artifact_directory=search_directory),
            artifact_directory=automl_directory,
        )
    return result
