"""After installing the built wheel, exercise it outside the source checkout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = r"""
import json
from pathlib import Path
import benchforge
from benchforge import load_benchmark_config, load_search_config, run_benchmark_suite, run_search
from benchforge.artifacts import verify
from benchforge.core.config import OutputConfig
from benchforge.catalog import Catalog
import sys
source = Path(sys.argv[1])
assert not Path(benchforge.__file__).resolve().is_relative_to(source)
config = load_benchmark_config(source / "configs/providers/benchmark.yaml").model_copy(
    update={"output": OutputConfig(directory=Path("published"))})
first = run_benchmark_suite(config, checkpoint_dir=Path("workspace"))
second = run_benchmark_suite(config, checkpoint_dir=Path("workspace"), resume=True)
assert len(second.checkpoint_reused) == 2
assert not first.failures
assert verify(first.artifact_directory, deep=True).passed
with Catalog(Path("catalog.sqlite")) as catalog:
    assert catalog.ingest(first.artifact_directory)
search = load_search_config(source / "configs/providers/search.yaml").model_copy(
    update={"output": OutputConfig(directory=Path("published"))})
result = run_search(search, checkpoint_dir=Path("search-workspace"))
recovered = run_search(search, checkpoint_dir=Path("search-workspace"), resume=True)
assert len(recovered.checkpoint_reused) == 2
assert not recovered.checkpoint_executed
assert recovered.fingerprint == result.fingerprint
assert not result.failures
assert verify(result.artifact_directory, deep=True).passed
print(json.dumps({"wheel": benchforge.__file__, "benchmark": first.fingerprint,
    "search": result.fingerprint, "reused": second.checkpoint_reused}))
"""

with tempfile.TemporaryDirectory(prefix="benchforge-wheel-smoke-") as temporary:
    subprocess.run([sys.executable, "-c", CODE, str(ROOT)], cwd=temporary, check=True)
