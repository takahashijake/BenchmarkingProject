"""BenchForge: reproducible classical machine-learning benchmarks."""

from benchforge._version import __version__
from benchforge.core.config import RunConfig, load_run_config
from benchforge.execution.runner import RunResult, run_benchmark

__all__ = ["RunConfig", "RunResult", "__version__", "load_run_config", "run_benchmark"]
