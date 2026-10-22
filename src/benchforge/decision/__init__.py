"""Leakage-safe, fixed binary calibration and decision analysis."""

from typing import TYPE_CHECKING, Any

from benchforge.decision.config import (
    CalibrationConfig,
    DecisionConfig,
    ThresholdConfig,
    load_decision_config,
)
from benchforge.decision.results import DecisionResult

if TYPE_CHECKING:
    from benchforge.decision.runner import DecisionAnalysisError, run_decision_analysis


def __getattr__(name: str) -> Any:
    if name in {"DecisionAnalysisError", "run_decision_analysis"}:
        from importlib import import_module

        value = getattr(import_module("benchforge.decision.runner"), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "CalibrationConfig",
    "DecisionAnalysisError",
    "DecisionConfig",
    "DecisionResult",
    "ThresholdConfig",
    "load_decision_config",
    "run_decision_analysis",
]
