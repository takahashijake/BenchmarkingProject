"""Generate small real experiments and verification-gated HTML examples.

Run from the repository root: python examples/reporting/demo.py --output-dir /tmp/bf-reports
Only experiment creation trains; generate_report() operates on persisted evidence.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from benchforge import (
    load_automl_config,
    load_benchmark_config,
    load_robustness_config,
    load_search_config,
    run_automl,
    run_benchmark_suite,
    run_robustness,
    run_search,
)
from benchforge.core.config import OutputConfig
from benchforge.reporting import VerificationError, generate_report

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "examples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reporting-demo"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reports = root / "reports"
    reports.mkdir()
    output = OutputConfig(directory=root / "evidence")

    # 1: binary fixed benchmark. A changed regularization setting supplies example 5.
    benchmark = load_benchmark_config(CONFIGS / "breast_cancer_suite.yaml")
    benchmark = benchmark.model_copy(
        update={
            "models": benchmark.models[:2],
            "split": benchmark.split.model_copy(update={"n_splits": 2}),
            "output": output,
        }
    )
    binary = run_benchmark_suite(benchmark)
    assert binary.artifact_directory is not None
    generate_report(binary.artifact_directory, reports / "binary-benchmark.html", deep=True)
    models = tuple(
        m.model_copy(update={"parameters": {**m.parameters, "C": 0.25}})
        if m.name == "logistic_regression"
        else m
        for m in benchmark.models
    )
    variant = run_benchmark_suite(benchmark.model_copy(update={"models": models}))
    assert variant.artifact_directory is not None
    generate_report(
        binary.artifact_directory,
        reports / "compatible-comparison.html",
        compare=variant.artifact_directory,
        deep=True,
    )

    # 2: multiclass nested search, with unbiased outer scores and separate final selection.
    search = load_search_config(CONFIGS / "iris_multiclass_search.yaml")
    search = search.model_copy(
        update={
            "models": tuple(m for m in search.models if m.id in {"logistic", "dummy"}),
            "output": output,
        }
    )
    nested = run_search(search)
    assert nested.artifact_directory is not None
    generate_report(nested.artifact_directory, reports / "multiclass-search.html", deep=True)

    # 3: regression AutoML; a small precommitted budget for one tuned family + dummy.
    automl = load_automl_config(CONFIGS / "diabetes_regression_automl.yaml")
    automl = automl.model_copy(
        update={
            "models": automl.models.model_copy(
                update={"include": ("ridge_regressor", "dummy_regressor")}
            ),
            "budget": automl.budget.model_copy(
                update={"total_trials": 4, "final_search_trials": 2}
            ),
            "output": output,
        }
    )
    regression = run_automl(automl)
    assert regression.artifact_directory is not None
    generate_report(regression.artifact_directory, reports / "regression-automl.html", deep=True)

    # 4: robustness pairwise comparison within one experiment, plus an experiment comparison.
    robustness = load_robustness_config(CONFIGS / "breast_cancer_robustness.yaml")
    robustness = robustness.model_copy(
        update={
            "models": robustness.models[:2],
            "split": robustness.split.model_copy(update={"n_splits": 2}),
            "robustness": robustness.robustness.model_copy(
                update={"repetitions": 3, "bootstrap_samples": 100}
            ),
            "output": output,
        }
    )
    repeated = run_robustness(robustness)
    assert repeated.artifact_directory is not None
    generate_report(repeated.artifact_directory, reports / "robustness.html", deep=True)
    # Same evidence copied to a second location gives a compatible comparison with zero deltas.
    replica = Path(shutil.copytree(repeated.artifact_directory, root / "robustness-replica"))
    generate_report(
        repeated.artifact_directory,
        reports / "robustness-comparison.html",
        compare=replica,
        deep=True,
    )

    # 6: deliberately damage expendable copies, preserving every original experiment.
    corrupt = Path(shutil.copytree(binary.artifact_directory, root / "corrupt-copy"))
    (corrupt / "metadata.json").write_text("{", encoding="utf-8")
    incomplete = root / "incomplete-copy"
    incomplete.mkdir()
    (incomplete / "search_config.json").write_text("{}", encoding="utf-8")
    for source in (corrupt, incomplete):
        try:
            generate_report(source, reports / f"{source.name}.html", deep=True)
        except VerificationError as exc:
            print(f"Expected rejection for {source.name}: {exc}")
        else:
            raise AssertionError("damaged evidence unexpectedly produced a report")
    print(f"Reports: {reports}")
    for report in sorted(reports.glob("*.html")):
        print(f"  {report.as_uri()}")


if __name__ == "__main__":
    main()
