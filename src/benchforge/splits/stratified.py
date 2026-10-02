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
