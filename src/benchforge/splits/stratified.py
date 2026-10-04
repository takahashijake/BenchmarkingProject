from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from benchforge.core.config import SplitConfig, TaskType


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]


def build_stratified_folds(
    target: pd.Series,
    config: SplitConfig,
    seed: int,
    task: TaskType = TaskType.BINARY_CLASSIFICATION,
) -> tuple[Fold, ...]:
    if config.strategy != "stratified_kfold":
        raise ValueError("classification requires split strategy 'stratified_kfold'")
    class_counts = target.value_counts()
    class_count = len(class_counts)
    if task == TaskType.BINARY_CLASSIFICATION and class_count != 2:
        raise ValueError("binary classification requires exactly two target classes")
    if task == TaskType.MULTICLASS_CLASSIFICATION and class_count < 3:
        raise ValueError("multiclass classification requires at least three target classes")
    if not task.is_classification:
        raise ValueError(f"stratified folds are incompatible with task '{task}'")
    if int(class_counts.min()) < config.n_splits:
        raise ValueError(
            f"each target class needs at least {config.n_splits} rows for "
            "stratified cross-validation"
        )
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_state=seed,
    )
    folds: list[Fold] = []
    for fold_id, (train, validation) in enumerate(splitter.split(np.zeros(len(target)), target)):
        train.setflags(write=False)
        validation.setflags(write=False)
        folds.append(Fold(fold_id, train, validation))
    return tuple(folds)


def build_kfold_folds(target: pd.Series, config: SplitConfig, seed: int) -> tuple[Fold, ...]:
    if config.strategy != "kfold":
        raise ValueError("regression requires split strategy 'kfold'")
    if len(target) < config.n_splits:
        raise ValueError(f"K-fold cross-validation needs at least {config.n_splits} rows")
    splitter = KFold(
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_state=seed,
    )
    folds: list[Fold] = []
    for fold_id, (train, validation) in enumerate(splitter.split(np.zeros(len(target)))):
        train.setflags(write=False)
        validation.setflags(write=False)
        folds.append(Fold(fold_id, train, validation))
    return tuple(folds)


def build_folds(
    target: pd.Series, config: SplitConfig, seed: int, task: TaskType
) -> tuple[Fold, ...]:
    if task.is_classification:
        return build_stratified_folds(target, config, seed, task)
    if task == TaskType.REGRESSION:
        return build_kfold_folds(target, config, seed)
    raise ValueError(f"unsupported task '{task}'")  # pragma: no cover


def build_inner_folds(
    target: pd.Series,
    outer_fold: Fold,
    config: SplitConfig,
    seed: int,
    task: TaskType = TaskType.BINARY_CLASSIFICATION,
) -> tuple[Fold, ...]:
    """Build inner folds in global index space, strictly inside an outer training set."""
    outer_train = outer_fold.train_indices
    relative_folds = build_folds(
        target.iloc[outer_train].reset_index(drop=True), config, seed, task
    )
    folds: list[Fold] = []
    for relative in relative_folds:
        train = np.asarray(outer_train[relative.train_indices], dtype=np.int64)
        validation = np.asarray(outer_train[relative.validation_indices], dtype=np.int64)
        train.setflags(write=False)
        validation.setflags(write=False)
        folds.append(Fold(relative.fold_id, train, validation))
    return tuple(folds)
