from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchforge.core.config import (
    AutoMLConfig,
    FinalSearchConfig,
    ModelConfig,
    SearchBudget,
    SearchConfig,
    SearchModelConfig,
)
from benchforge.data.registry import Dataset, DatasetRegistry, default_dataset_registry
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.search.registry import SearchSpaceRegistry, default_search_space_registry


@dataclass(frozen=True)
class AutoMLPlan:
    fingerprint: str
    dataset_identity: str
    searchable_families: tuple[str, ...]
    fixed_families: tuple[str, ...]
    candidates: tuple[SearchModelConfig, ...]
    outer_fold_count: int
    inner_fold_count: int
    total_trial_budget: int
    final_search_trials: int
    outer_search_budget: int
    study_count: int
    trials_per_outer_fold: int
    allocated_outer_trials: int
    allocated_trials: int
    unallocated_trials: int
    approximate_maximum_fits: int
    search_fingerprint: str
    search_config: SearchConfig
    selected_model_identity: dict[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "dataset_identity": self.dataset_identity,
            "searchable_families": list(self.searchable_families),
            "fixed_families": list(self.fixed_families),
            "candidates": [
                candidate.model_dump(mode="json", exclude_none=True)
                for candidate in self.candidates
            ],
            "outer_fold_count": self.outer_fold_count,
            "inner_fold_count": self.inner_fold_count,
            "total_trial_budget": self.total_trial_budget,
            "final_search_trials": self.final_search_trials,
            "outer_search_budget": self.outer_search_budget,
            "study_count": self.study_count,
            "trials_per_outer_fold": self.trials_per_outer_fold,
            "allocated_outer_trials": self.allocated_outer_trials,
            "allocated_trials": self.allocated_trials,
            "unallocated_trials": self.unallocated_trials,
            "approximate_maximum_fits": self.approximate_maximum_fits,
            "search_fingerprint": self.search_fingerprint,
        }


def build_automl_plan(
    config: AutoMLConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    search_space_registry: SearchSpaceRegistry = default_search_space_registry,
    dataset: Dataset | None = None,
) -> AutoMLPlan:
    """Resolve policy into a deterministic ordinary nested-search configuration."""
    resolved_dataset = dataset or dataset_registry.resolve(config.dataset)
    if resolved_dataset.task != config.task:
        raise ValueError(
            f"dataset task '{resolved_dataset.task}' does not match configured task '{config.task}'"
        )

    requested = set(config.models.include) | set(config.models.exclude)
    known = set(model_registry.names)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown AutoML model names: {', '.join(unknown)}")
    incompatible = sorted(
        name for name in requested if not model_registry.supports_task(name, config.task)
    )
    if incompatible:
        raise ValueError(
            f"models incompatible with task '{config.task}': {', '.join(incompatible)}"
        )

    compatible = {
        name for name in model_registry.names if model_registry.supports_task(name, config.task)
    }
    selected = set(config.models.include) if config.models.include else compatible
    selected -= set(config.models.exclude)
    searchable_names = set(search_space_registry.names)
    searchable = tuple(sorted(selected & searchable_names))
    fixed_selected = selected - searchable_names
    fixed = tuple(sorted(fixed_selected)) if config.models.include_fixed_baselines else ()
    if not searchable:
        raise ValueError("AutoML policy selects no searchable model families")

    outer_budget = config.budget.total_trials - config.budget.final_search_trials
    study_count = len(searchable) * config.outer_split.n_splits
    trials_per_study, remainder = divmod(outer_budget, study_count)
    if trials_per_study < 1:
        raise ValueError(
            "AutoML trial budget is too small: "
            f"{outer_budget} outer-search trials cannot fund {study_count} "
            "searchable-family/outer-fold studies"
        )
    allocated = trials_per_study * study_count
    allocated_total = allocated + config.budget.final_search_trials

    candidates = tuple(
        [SearchModelConfig(name=name, mode="search") for name in searchable]
        + [SearchModelConfig(name=name, mode="fixed") for name in fixed]
    )
    for candidate in candidates:
        parameters = (
            search_space_registry.resolve(candidate.name).defaults
            if candidate.mode == "search"
            else {}
        )
        model_registry.create(
            ModelConfig(name=candidate.name, parameters=parameters),
            config.seed,
            config.task,
        )

    final_trials = config.budget.final_search_trials
    generated = SearchConfig(
        task=config.task,
        dataset=config.dataset,
        outer_split=config.outer_split,
        inner_split=config.inner_split,
        preprocessing=config.preprocessing,
        models=candidates,
        metrics=config.metrics,
        primary_metric=config.primary_metric,
        budget=SearchBudget(
            trials_per_outer_fold=trials_per_study,
            timeout_seconds_per_outer_fold=config.budget.timeout_seconds_per_outer_fold,
        ),
        final_search=FinalSearchConfig(
            enabled=final_trials > 0,
            trials=final_trials,
        ),
        seed=config.seed,
        output=config.output,
    )
    space_identity = search_space_registry.identity_for(set(searchable))
    search_fingerprint = generated.fingerprint_for_dataset(
        resolved_dataset.identity, space_identity
    )
    fit_count = (
        allocated * config.inner_split.n_splits
        + len(candidates) * config.outer_split.n_splits
        + final_trials * config.inner_split.n_splits
    )
    allocation = {
        "total_trial_budget": config.budget.total_trials,
        "final_search_trials": final_trials,
        "outer_search_budget": outer_budget,
        "study_count": study_count,
        "trials_per_outer_fold": trials_per_study,
        "allocated_outer_trials": allocated,
        "allocated_trials": allocated_total,
        "unallocated_trials": remainder,
        "approximate_maximum_fits": fit_count,
    }
    selected_identity = model_registry.identity_for(set((*searchable, *fixed)))
    selected_identity = {
        name: {
            **identity,
            "mode": "search" if name in searchable else "fixed",
        }
        for name, identity in selected_identity.items()
    }
    fingerprint = config.fingerprint_for_plan(
        dataset_identity=resolved_dataset.identity,
        selected_models=selected_identity,
        search_spaces=space_identity,
        allocation=allocation,
        search_fingerprint=search_fingerprint,
    )
    return AutoMLPlan(
        fingerprint=fingerprint,
        dataset_identity=resolved_dataset.identity,
        searchable_families=searchable,
        fixed_families=fixed,
        candidates=candidates,
        outer_fold_count=config.outer_split.n_splits,
        inner_fold_count=config.inner_split.n_splits,
        total_trial_budget=config.budget.total_trials,
        final_search_trials=final_trials,
        outer_search_budget=outer_budget,
        study_count=study_count,
        trials_per_outer_fold=trials_per_study,
        allocated_outer_trials=allocated,
        allocated_trials=allocated_total,
        unallocated_trials=remainder,
        approximate_maximum_fits=fit_count,
        search_fingerprint=search_fingerprint,
        search_config=generated,
        selected_model_identity=selected_identity,
    )
