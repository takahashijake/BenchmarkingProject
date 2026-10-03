"""BenchForge: reproducible classical machine-learning benchmarks."""

from benchforge._version import __version__
from benchforge.core.config import (
    BenchmarkConfig,
    RunConfig,
    load_benchmark_config,
    load_run_config,
)
from benchforge.execution.benchmark import BenchmarkResult, run_benchmark_suite
from benchforge.execution.runner import RunResult, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "RunConfig",
    "RunResult",
    "__version__",
    "load_benchmark_config",
    "load_run_config",
    "run_benchmark",
    "run_benchmark_suite",
]
