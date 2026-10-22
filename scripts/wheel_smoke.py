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
from benchforge.reporting import generate_report, load_report
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
parallel = load_search_config(source / "configs/examples/breast_cancer_search.yaml")
parallel = parallel.model_copy(update={
    "models": tuple(m for m in parallel.models if m.id in {"logistic", "dummy"}),
    "budget": parallel.budget.model_copy(update={"trials_per_outer_fold": 1}),
    "final_search": parallel.final_search.model_copy(update={"enabled": False}),
    "output": OutputConfig(directory=Path("published")),
})
serial = run_search(parallel, persist=False)
isolated = run_search(parallel, workers=2, checkpoint_dir=Path("parallel-workspace"))
resumed = run_search(parallel, workers=1, checkpoint_dir=Path("parallel-workspace"), resume=True)
assert serial.fingerprint == isolated.fingerprint == resumed.fingerprint
assert len(resumed.checkpoint_reused) == 4
assert [f.predictions for f in serial.families] == [f.predictions for f in isolated.families]
assert ([f.aggregate_metrics for f in serial.families]
    == [f.aggregate_metrics for f in isolated.families])
assert verify(isolated.artifact_directory, deep=True).passed
assert verify(resumed.artifact_directory, deep=True).passed
report = generate_report(first.artifact_directory, Path("benchmark.html"), deep=True)
comparison = generate_report(isolated.artifact_directory, Path("comparison.html"),
    compare=resumed.artifact_directory, deep=True)
assert load_report(first.artifact_directory).verification.integrity == "sealed"
for html_path in (report, comparison):
    html = html_path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and html.endswith("</html>\n")
    assert "<script" not in html and "<link" not in html and "@import" not in html
    assert "<table>" in html and "Content-Security-Policy" in html
assert "COMPATIBLE" in comparison.read_text(encoding="utf-8")
print(json.dumps({"wheel": benchforge.__file__, "benchmark": first.fingerprint,
    "search": result.fingerprint, "reused": second.checkpoint_reused}))
"""

with tempfile.TemporaryDirectory(prefix="benchforge-wheel-smoke-") as temporary:
    subprocess.run([sys.executable, "-c", CODE, str(ROOT)], cwd=temporary, check=True)
