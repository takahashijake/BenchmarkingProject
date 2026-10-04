from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from benchforge.core.config import RunConfig, TaskType
from benchforge.data.registry import (
    Dataset,
    DatasetRegistry,
    DatasetSummary,
    default_dataset_registry,
    summarize_dataset,
)
from benchforge.evaluation.metrics import (
    AggregateMetric,
    aggregate_fold_metrics,
    compute_metrics,
    metric_spec,
)
from benchforge.models.registry import ModelRegistry, default_model_registry
from benchforge.preprocessing.builder import build_preprocessor
from benchforge.splits.stratified import Fold, build_folds
from benchforge.storage.local import LocalArtifactStore


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_size: int
    validation_size: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictionRecord:
    sample_index: int
    fold_id: int
    truth: int | float
    prediction: int | float
    score: float | None


@dataclass(frozen=True)
class RunResult:
    fingerprint: str
    dataset_identity: str
    dataset: DatasetSummary
    model_name: str
    seed: int
    fold_results: tuple[FoldResult, ...]
    aggregate_metrics: dict[str, AggregateMetric]
    predictions: tuple[PredictionRecord, ...]
    artifact_directory: Path | None = None


class FoldExecutionError(RuntimeError):
    pass


def _prediction_score(pipeline: Pipeline, features: Any) -> np.ndarray[Any, Any] | None:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        classes = pipeline.classes_
        positive_positions = np.flatnonzero(classes == 1)
        if len(positive_positions) != 1:
            raise ValueError("binary classification scoring requires a class labeled 1")
        return np.asarray(probabilities[:, positive_positions[0]], dtype=float)
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(features), dtype=float)
    return None


def execute_fold(
    config: RunConfig,
    dataset: Dataset,
    fold: Fold,
    *,
    model_registry: ModelRegistry = default_model_registry,
) -> tuple[FoldResult, tuple[PredictionRecord, ...]]:
    """Execute one immutable fold using the same kernel as a normal run."""
    try:
        estimator = model_registry.create(config.model, config.seed, config.task)
        capabilities = model_registry.capabilities(config.model.name)
        pipeline = Pipeline(
            [
                (
                    "preprocessing",
                    build_preprocessor(
                        config.preprocessing,
                        dataset.numeric_feature_names,
                        dataset.categorical_feature_names,
                        dense_output=capabilities.requires_dense,
                    ),
                ),
                ("model", estimator),
            ]
        )
        train_x = dataset.features.iloc[fold.train_indices]
        train_y = dataset.target.iloc[fold.train_indices]
        validation_x = dataset.features.iloc[fold.validation_indices]
        validation_y = dataset.target.iloc[fold.validation_indices]
        pipeline.fit(train_x, train_y)
        if config.task.is_classification:
            predicted = np.asarray(pipeline.predict(validation_x), dtype=np.int64)
            truth = validation_y.to_numpy(dtype=np.int64)
            score = (
                _prediction_score(pipeline, validation_x)
                if any(metric_spec(metric).requires_score for metric in config.metrics)
                else None
            )
        elif config.task == TaskType.REGRESSION:
            predicted = np.asarray(pipeline.predict(validation_x), dtype=np.float64)
            truth = validation_y.to_numpy(dtype=np.float64)
            score = None
        else:  # pragma: no cover - TaskType validation is exhaustive
            raise ValueError(f"unsupported task '{config.task}'")
        metrics = compute_metrics(config.metrics, truth, predicted, score, config.task)
        fold_result = FoldResult(fold.fold_id, len(train_x), len(validation_x), metrics)
        predictions = tuple(
            PredictionRecord(
                sample_index=int(sample_index),
                fold_id=fold.fold_id,
                truth=(
                    int(actual) if config.task.is_classification else float(actual)
                ),
                prediction=(
                    int(estimate)
                    if config.task.is_classification
                    else float(estimate)
                ),
                score=None if score is None else float(score[position]),
            )
            for position, (sample_index, actual, estimate) in enumerate(
                zip(validation_y.index, truth, predicted, strict=True)
            )
        )
        return fold_result, predictions
    except Exception as exc:
        raise FoldExecutionError(
            f"fold {fold.fold_id} failed for dataset '{dataset.identity}', "
            f"model '{config.model.name}', fingerprint '{config.fingerprint[:12]}': {exc}"
        ) from exc


def run_benchmark(
    config: RunConfig,
    *,
    dataset_registry: DatasetRegistry = default_dataset_registry,
    model_registry: ModelRegistry = default_model_registry,
    persist: bool = True,
    dataset: Dataset | None = None,
    folds: tuple[Fold, ...] | None = None,
) -> RunResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    if dataset is None:
        dataset = dataset_registry.resolve(config.dataset)
    if dataset.task != config.task:
        raise ValueError(
            f"dataset task '{dataset.task}' does not match configured task '{config.task}'"
        )
    if folds is None:
        folds = build_folds(dataset.target, config.split, config.seed, config.task)
    if len(folds) != config.split.n_splits:
        raise ValueError(
            f"provided fold plan has {len(folds)} folds; expected {config.split.n_splits}"
        )
    fold_results: list[FoldResult] = []
    predictions: list[PredictionRecord] = []

    for fold in folds:
        fold_result, fold_predictions = execute_fold(
            config, dataset, fold, model_registry=model_registry
        )
        fold_results.append(fold_result)
        predictions.extend(fold_predictions)

    result = RunResult(
        fingerprint=config.fingerprint_for_dataset(dataset.identity),
        dataset_identity=dataset.identity,
        dataset=summarize_dataset(dataset),
        model_name=config.model.name,
        seed=config.seed,
        fold_results=tuple(fold_results),
        aggregate_metrics=aggregate_fold_metrics(item.metrics for item in fold_results),
        predictions=tuple(sorted(predictions, key=lambda item: item.sample_index)),
    )
    if persist:
        directory = LocalArtifactStore(config.output.directory).write(config, result)
        result = replace(result, artifact_directory=directory)
    return result
