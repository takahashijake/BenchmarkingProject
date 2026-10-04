"""Deterministic repetition policy; ordinary split semantics live in splits."""

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
import pydantic
import sklearn

from benchforge._version import __version__
from benchforge.core.seeds import derive_seed
from benchforge.data.registry import Dataset, DatasetRegistry, default_dataset_registry
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.robustness.config import RobustnessConfig
from benchforge.splits.stratified import Fold, build_folds

METHODOLOGY_VERSION = "matched-cv-v1:sha256-seeds:competition-ranks:sample-std:percentile-bootstrap"


def dependency_versions() -> dict[str, str]:
    return {
        "benchforge": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pydantic": pydantic.__version__,
        "scipy": version("scipy"),
        "pyarrow": version("pyarrow"),
    }


@dataclass(frozen=True)
class RepetitionPlan:
    repetition_id: int
    seed: int
    folds: tuple[Fold, ...]


@dataclass(frozen=True)
class RobustnessPlan:
    fingerprint: str
    candidate_identifiers: tuple[str, ...]
    repetitions: tuple[RepetitionPlan, ...]
    approximate_maximum_fits: int
    methodology_version: str = METHODOLOGY_VERSION


def build_robustness_plan(
    config: RobustnessConfig,
    dataset: Dataset | None = None,
    model_registry: ModelRegistry = default_model_registry,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
) -> RobustnessPlan:
    if dataset is None:
        dataset = dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError("dataset task must match the robustness task")
    for model in config.models:
        if not model_registry.supports_task(model.name, config.task):
            raise ValueError(f"model '{model.name}' is incompatible with task '{config.task}'")
    fingerprint = robustness_fingerprint(config, dataset.identity, model_registry)
    repetitions = []
    used_seeds: set[int] = set()
    for repetition_id in range(config.robustness.repetitions):
        nonce = 0
        seed = derive_seed(config.seed, "robustness", "folds", repetition_id, nonce)
        # Resolve the rare uint32 collision deterministically, without silently reusing a seed.
        while seed in used_seeds:
            nonce += 1
            seed = derive_seed(config.seed, "robustness", "folds", repetition_id, nonce)
        used_seeds.add(seed)
        repetitions.append(
            RepetitionPlan(
                repetition_id, seed, build_folds(dataset.target, config.split, seed, config.task)
            )
        )
    return RobustnessPlan(
        fingerprint,
        tuple(sorted(model.id or model.name for model in config.models)),
        tuple(repetitions),
        config.robustness.repetitions * len(config.models) * config.split.n_splits,
    )


def robustness_fingerprint(
    config: RobustnessConfig,
    dataset_identity: str,
    model_registry: ModelRegistry = default_model_registry,
) -> str:
    semantics = config.canonical_dict(include_output=False)
    semantics["dataset"].pop("path", None)
    identity = {
        "robustness": semantics,
        "dataset_identity": dataset_identity,
        "models": model_registry.identity_for({model.name for model in config.models}),
        "versions": dependency_versions(),
        "methodology_version": METHODOLOGY_VERSION,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
