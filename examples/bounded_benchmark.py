"""Run from the repository: python examples/bounded_benchmark.py."""
from pathlib import Path
from tempfile import mkdtemp

from benchforge.artifacts.verify import verify
from benchforge.catalog import Catalog
from benchforge.core.config import OutputConfig, load_benchmark_config
from benchforge.execution import ExecutionOptions, run_benchmark_suite

root = Path(mkdtemp(prefix="benchforge-v011-demo-"))
config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
config = config.model_copy(update={
    "models": config.models[:2], "output": OutputConfig(directory=root / "published"),
})
normal = run_benchmark_suite(config, workers=2, execution_options=ExecutionOptions(30, 90))
assert not normal.failures
stopped = run_benchmark_suite(
    config, checkpoint_dir=root / "work", execution_options=ExecutionOptions(run_timeout=0.001),
)
assert stopped.failures and not stopped.candidates
print("Budget stopped:", [(f.model_identifier, f.error_type) for f in stopped.failures])
resumed = run_benchmark_suite(
    config, checkpoint_dir=root / "work", resume=True, workers=2,
    execution_options=ExecutionOptions(30, 90),
)
assert not resumed.failures and resumed.artifact_directory is not None
assert verify(resumed.artifact_directory, deep=True).passed
with Catalog(root / "catalog.sqlite") as catalog:
    assert len(catalog.ingest(resumed.artifact_directory)) == 1
print("Verified resumed artifact:", resumed.artifact_directory)
