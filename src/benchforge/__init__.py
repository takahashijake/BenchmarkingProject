"""BenchForge: reproducible classical machine-learning benchmarks."""

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
from benchforge.execution.benchmark import BenchmarkResult, run_benchmark_suite
from benchforge.execution.runner import RunResult, run_benchmark
from benchforge.robustness import (
    RobustnessConfig,
    RobustnessResult,
    load_robustness_config,
    run_robustness,
)
from benchforge.search import SearchResult, run_search

__all__ = [
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
