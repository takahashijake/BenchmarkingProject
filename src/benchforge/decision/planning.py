"""Training-only fold plans, identity snapshots and explicit seed derivation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from benchforge.core.config import SplitConfig
from benchforge.core.seeds import derive_seed
from benchforge.decision.config import DecisionConfig
from benchforge.decision.results import (
    DatasetEvidence,
    DecisionPlan,
    FitBoundary,
    OuterPlan,
    Partition,
    SplitPlan,
)
from benchforge.splits.stratified import Fold, build_folds, build_inner_folds

METHODOLOGY_VERSION = "binary-decision-v1"
SEMANTICS = {
    "methodology": METHODOLOGY_VERSION,
    "seeds": "sha256-json-uint32:decision:outer|inner|calibration|estimator",
    "calibration": "sklearn.CalibratedClassifierCV:ensemble=False:n_jobs=1:full-pipeline-cv",
    "calibration_response": "decision_function preferred, otherwise predict_proba",
    "positive_class": 1,
    "prediction": "probability >= threshold",
    "selection": "inner-oof-only:exact-rational-objective:decimal-distance-to-0.5:lower-threshold",
    "cost": "(false_positive_cost*FP + false_negative_cost*FN)/N",
    "probability": "Brier=mean((p-y)^2);log_loss=sklearn-float64-eps;AUC=positive-1",
    "reliability": "uniform-linspace:left-closed-right-open:last-inclusive:empty-null",
    "ece": "sum(bin_count/N * abs(mean_probability-positive_fraction))",
    "aggregation": "unweighted-outer-fold-mean:population-std-ddof-0:maxabs-rescaled-ge1",
    "ranking": "complete-only:outer-primary:exact-competition-ties",
}


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def fingerprint(identity_json: str) -> str:
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()


def dataset_content_identity(dataset: DatasetEvidence) -> str:
    """Include the original label meanings, not just their encoded integer values."""
    return "sha256:" + fingerprint(
        canonical(
            {
                "content_digest": dataset.content_digest,
                "target_digest": dataset.target_digest,
                "target_labels": dataset.target_labels,
            }
        )
    )


def estimator_seed(config: DecisionConfig, identifier: str, outer: int, inner: int | None) -> int:
    return derive_seed(config.seed, "decision", "estimator", identifier, outer, inner)


def partition(fold: Fold) -> Partition:
    return Partition(
        fold_id=fold.fold_id,
        train_indices=tuple(map(int, fold.train_indices)),
        validation_indices=tuple(map(int, fold.validation_indices)),
    )


def as_fold(part: Partition) -> Fold:
    return Fold(
        part.fold_id,
        np.asarray(part.train_indices, dtype=np.int64),
        np.asarray(part.validation_indices, dtype=np.int64),
    )


def build_plan(config: DecisionConfig, truth: tuple[int, ...]) -> DecisionPlan:
    """Preflight all nesting layers before training; indices remain dataset positions."""
    target = pd.Series(truth, dtype=np.int64)
    outer_seed = derive_seed(config.seed, "decision", "outer")
    outer = build_folds(target, config.outer_split, outer_seed, config.task)

    def boundary(fold: Fold, outer_id: int, inner_id: int | None) -> FitBoundary:
        if config.calibration.method == "none":
            return FitBoundary(fold=partition(fold), calibration=None)
        if (
            config.calibration.method == "isotonic"
            and len(fold.train_indices) < config.calibration.isotonic_min_samples
        ):
            raise ValueError(
                f"outer fold {outer_id}, inner {inner_id}: isotonic requires at least "
                f"{config.calibration.isotonic_min_samples} training-only calibration samples; "
                f"found {len(fold.train_indices)}"
            )
        seed = derive_seed(config.seed, "decision", "calibration", outer_id, inner_id)
        try:
            folds = build_inner_folds(
                target, fold, SplitConfig(n_splits=config.calibration.n_splits), seed
            )
        except ValueError as exc:
            raise ValueError(
                f"outer fold {outer_id}, inner {inner_id}: calibration: {exc}"
            ) from exc
        return FitBoundary(
            fold=partition(fold),
            calibration=SplitPlan(seed=seed, folds=tuple(partition(f) for f in folds)),
        )

    plans = []
    for fold in outer:
        seed = derive_seed(config.seed, "decision", "inner", fold.fold_id)
        try:
            inner = build_inner_folds(target, fold, config.inner_split, seed)
        except ValueError as exc:
            raise ValueError(f"outer fold {fold.fold_id}: inner: {exc}") from exc
        plans.append(
            OuterPlan(
                final=boundary(fold, fold.fold_id, None),
                inner_seed=seed,
                inner=tuple(boundary(f, fold.fold_id, f.fold_id) for f in inner),
            )
        )
    return DecisionPlan(outer_seed=outer_seed, outer=tuple(plans))
