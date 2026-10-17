"""Typed public API, loaded on demand to keep worker startup lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchforge.execution.benchmark import BenchmarkResult, run_benchmark_suite
    from benchforge.execution.runner import RunResult, run_benchmark
    from benchforge.execution.supervisor import ExecutionOptions

_EXPORTS = {
    "BenchmarkResult": ("benchforge.execution.benchmark", "BenchmarkResult"),
    "run_benchmark_suite": ("benchforge.execution.benchmark", "run_benchmark_suite"),
    "RunResult": ("benchforge.execution.runner", "RunResult"),
    "run_benchmark": ("benchforge.execution.runner", "run_benchmark"),
    "ExecutionOptions": ("benchforge.execution.supervisor", "ExecutionOptions"),
}

__all__ = [
    "ExecutionOptions",
    "BenchmarkResult",
    "RunResult",
    "run_benchmark",
    "run_benchmark_suite",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
