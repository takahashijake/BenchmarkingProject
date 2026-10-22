"""Typed public API, loaded on demand to keep worker startup lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchforge._version import __version__
    from benchforge.automl import AutoMLPlan, AutoMLResult, build_automl_plan, run_automl
    from benchforge.core.config import (
        AutoMLConfig,
        BenchmarkConfig,
        RunConfig,
        SearchConfig,
        load_automl_config,
        load_benchmark_config,
        load_run_config,
        load_search_config,
    )
    from benchforge.decision import (
        DecisionConfig,
        DecisionResult,
        load_decision_config,
        run_decision_analysis,
    )
    from benchforge.execution.benchmark import BenchmarkResult, run_benchmark_suite
    from benchforge.execution.runner import RunResult, run_benchmark
    from benchforge.robustness import (
        RobustnessConfig,
        RobustnessResult,
        load_robustness_config,
        run_robustness,
    )
    from benchforge.search import SearchResult, run_search

_EXPORTS = {
    "DecisionConfig": ("benchforge.decision", "DecisionConfig"),
    "DecisionResult": ("benchforge.decision", "DecisionResult"),
    "load_decision_config": ("benchforge.decision", "load_decision_config"),
    "run_decision_analysis": ("benchforge.decision", "run_decision_analysis"),
    "__version__": ("benchforge._version", "__version__"),
    "AutoMLPlan": ("benchforge.automl", "AutoMLPlan"),
    "AutoMLResult": ("benchforge.automl", "AutoMLResult"),
    "build_automl_plan": ("benchforge.automl", "build_automl_plan"),
    "run_automl": ("benchforge.automl", "run_automl"),
    "AutoMLConfig": ("benchforge.core.config", "AutoMLConfig"),
    "BenchmarkConfig": ("benchforge.core.config", "BenchmarkConfig"),
    "RunConfig": ("benchforge.core.config", "RunConfig"),
    "SearchConfig": ("benchforge.core.config", "SearchConfig"),
    "load_automl_config": ("benchforge.core.config", "load_automl_config"),
    "load_benchmark_config": ("benchforge.core.config", "load_benchmark_config"),
    "load_run_config": ("benchforge.core.config", "load_run_config"),
    "load_search_config": ("benchforge.core.config", "load_search_config"),
    "BenchmarkResult": ("benchforge.execution.benchmark", "BenchmarkResult"),
    "run_benchmark_suite": ("benchforge.execution.benchmark", "run_benchmark_suite"),
    "RunResult": ("benchforge.execution.runner", "RunResult"),
    "run_benchmark": ("benchforge.execution.runner", "run_benchmark"),
    "RobustnessConfig": ("benchforge.robustness", "RobustnessConfig"),
    "RobustnessResult": ("benchforge.robustness", "RobustnessResult"),
    "load_robustness_config": ("benchforge.robustness", "load_robustness_config"),
    "run_robustness": ("benchforge.robustness", "run_robustness"),
    "SearchResult": ("benchforge.search", "SearchResult"),
    "run_search": ("benchforge.search", "run_search"),
}

__all__ = [
    "DecisionConfig",
    "DecisionResult",
    "load_decision_config",
    "run_decision_analysis",
    "AutoMLConfig",
    "AutoMLPlan",
    "AutoMLResult",
    "BenchmarkConfig",
    "BenchmarkResult",
    "RobustnessConfig",
    "RobustnessResult",
    "load_robustness_config",
    "run_robustness",
    "RunConfig",
    "RunResult",
    "SearchConfig",
    "SearchResult",
    "__version__",
    "build_automl_plan",
    "load_automl_config",
    "load_benchmark_config",
    "load_run_config",
    "load_search_config",
    "run_benchmark",
    "run_benchmark_suite",
    "run_automl",
    "run_search",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
