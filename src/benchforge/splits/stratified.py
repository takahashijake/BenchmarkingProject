from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from benchforge.core.config import SplitConfig


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]


def build_stratified_folds(target: pd.Series, config: SplitConfig, seed: int) -> tuple[Fold, ...]:
    class_counts = target.value_counts()
    if len(class_counts) != 2:
        raise ValueError("stratified binary cross-validation requires exactly two target classes")
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


def build_inner_folds(
    target: pd.Series,
    outer_fold: Fold,
    config: SplitConfig,
    seed: int,
) -> tuple[Fold, ...]:
    """Build inner folds in global index space, strictly inside an outer training set."""
    outer_train = outer_fold.train_indices
    relative_folds = build_stratified_folds(
        target.iloc[outer_train].reset_index(drop=True), config, seed
    )
    folds: list[Fold] = []
    for relative in relative_folds:
        train = np.asarray(outer_train[relative.train_indices], dtype=np.int64)
        validation = np.asarray(outer_train[relative.validation_indices], dtype=np.int64)
        train.setflags(write=False)
        validation.setflags(write=False)
        folds.append(Fold(relative.fold_id, train, validation))
    return tuple(folds)
