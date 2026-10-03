from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import optuna

from benchforge.core.config import JsonScalar

SuggestFunction = Callable[[optuna.trial.Trial, dict[str, JsonScalar]], dict[str, JsonScalar]]


@dataclass(frozen=True)
class SearchSpace:
    name: str
    version: int
    definition: dict[str, Any]
    suggest: SuggestFunction
    defaults: dict[str, JsonScalar] = field(default_factory=dict)

    @property
    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "definition": self.definition}

    def propose(
        self, trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]
    ) -> dict[str, JsonScalar]:
        """Propose only unfixed parameters; user-fixed values always win deliberately."""
        proposed = self.suggest(trial, fixed)
        overlap = proposed.keys() & fixed.keys()
        if overlap:  # defensive: every built-in proposer filters fixed names
            names = ", ".join(sorted(overlap))
            raise ValueError(f"search space attempted to overwrite fixed parameters: {names}")
        return proposed


def _logistic(trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
    values: dict[str, JsonScalar] = {}
    if "C" not in fixed:
        values["C"] = trial.suggest_float("C", 1e-4, 1e3, log=True)
    if "penalty" not in fixed:
        solver = str(fixed.get("solver", "liblinear"))
        penalties = ["l2"] if solver in {"lbfgs", "newton-cg", "newton-cholesky"} else ["l1", "l2"]
        values["penalty"] = trial.suggest_categorical("penalty", penalties)
    if "class_weight" not in fixed:
        values["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
    return values


def _tree(trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
    values: dict[str, JsonScalar] = {}
    if "n_estimators" not in fixed:
        values["n_estimators"] = trial.suggest_int("n_estimators", 50, 300, step=50)
    if "max_depth" not in fixed:
        values["max_depth"] = trial.suggest_categorical("max_depth", [None, 4, 8, 16, 24])
    if "min_samples_split" not in fixed:
        values["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
    if "min_samples_leaf" not in fixed:
        values["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
    if "max_features" not in fixed:
        values["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    if "class_weight" not in fixed:
        values["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
    return values


def _ridge(trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
    if "alpha" in fixed:
        return {}
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}


def _regression_tree(
    trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]
) -> dict[str, JsonScalar]:
    values: dict[str, JsonScalar] = {}
    if "n_estimators" not in fixed:
        values["n_estimators"] = trial.suggest_int("n_estimators", 50, 300, step=50)
    if "max_depth" not in fixed:
        values["max_depth"] = trial.suggest_categorical("max_depth", [None, 4, 8, 16, 24])
    if "min_samples_split" not in fixed:
        values["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
    if "min_samples_leaf" not in fixed:
        values["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
    if "max_features" not in fixed:
        values["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0])
    return values


def _hist_gradient_boosting(
    trial: optuna.trial.Trial, fixed: dict[str, JsonScalar]
) -> dict[str, JsonScalar]:
    values: dict[str, JsonScalar] = {}
    if "learning_rate" not in fixed:
        values["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    if "max_iter" not in fixed:
        values["max_iter"] = trial.suggest_int("max_iter", 50, 250, step=25)
    if "max_leaf_nodes" not in fixed:
        values["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 7, 63)
    if "min_samples_leaf" not in fixed:
        values["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 40)
    if "l2_regularization" not in fixed:
        values["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)
    if "max_depth" not in fixed:
        values["max_depth"] = trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12])
    return values


LOGISTIC_SPACE = SearchSpace(
    name="logistic_regression",
    version=1,
    definition={
        "C": {"type": "float", "low": 1e-4, "high": 1e3, "log": True},
        "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
        "defaults": {"solver": "liblinear", "max_iter": 1000},
    },
    suggest=_logistic,
    defaults={"solver": "liblinear", "max_iter": 1000},
)

TREE_DEFINITION = {
    "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
    "max_depth": {"type": "categorical", "choices": [None, 4, 8, 16, 24]},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
}

RANDOM_FOREST_SPACE = SearchSpace("random_forest_classifier", 1, TREE_DEFINITION, _tree)
EXTRA_TREES_SPACE = SearchSpace("extra_trees_classifier", 1, TREE_DEFINITION, _tree)
HIST_GRADIENT_BOOSTING_SPACE = SearchSpace(
    name="hist_gradient_boosting_classifier",
    version=1,
    definition={
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_iter": {"type": "int", "low": 50, "high": 250, "step": 25},
        "max_leaf_nodes": {"type": "int", "low": 7, "high": 63},
        "min_samples_leaf": {"type": "int", "low": 5, "high": 40},
        "l2_regularization": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "max_depth": {"type": "categorical", "choices": [None, 3, 5, 8, 12]},
    },
    suggest=_hist_gradient_boosting,
)

RIDGE_REGRESSOR_SPACE = SearchSpace(
    name="ridge_regressor",
    version=1,
    definition={"alpha": {"type": "float", "low": 1e-4, "high": 1e4, "log": True}},
    suggest=_ridge,
)

REGRESSION_TREE_DEFINITION = {
    "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
    "max_depth": {"type": "categorical", "choices": [None, 4, 8, 16, 24]},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 1.0]},
}

RANDOM_FOREST_REGRESSOR_SPACE = SearchSpace(
    "random_forest_regressor", 1, dict(REGRESSION_TREE_DEFINITION), _regression_tree
)
EXTRA_TREES_REGRESSOR_SPACE = SearchSpace(
    "extra_trees_regressor", 1, dict(REGRESSION_TREE_DEFINITION), _regression_tree
)
HIST_GRADIENT_BOOSTING_REGRESSOR_SPACE = SearchSpace(
    name="hist_gradient_boosting_regressor",
    version=1,
    definition={
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_iter": {"type": "int", "low": 50, "high": 250, "step": 25},
        "max_leaf_nodes": {"type": "int", "low": 7, "high": 63},
        "min_samples_leaf": {"type": "int", "low": 5, "high": 40},
        "l2_regularization": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "max_depth": {"type": "categorical", "choices": [None, 3, 5, 8, 12]},
    },
    suggest=_hist_gradient_boosting,
)
