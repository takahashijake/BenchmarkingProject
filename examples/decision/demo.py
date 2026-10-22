"""Run both complete examples, reproduce evidence and render verified static HTML."""

import argparse
import json
from pathlib import Path

from benchforge import DecisionConfig, load_decision_config, run_decision_analysis
from benchforge.analysis.decision import format_decision_summary
from benchforge.artifacts import verify
from benchforge.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.absolute()
    root.mkdir(parents=True, exist_ok=False)
    (root / "reports").mkdir()
    source = Path(__file__).resolve().parents[2]
    audit = {}
    for label, filename in (
        ("f1", "breast_cancer_decision.yaml"),
        ("asymmetric-cost", "breast_cancer_decision_cost.yaml"),
    ):
        config = load_decision_config(source / "configs/examples" / filename)
        config = DecisionConfig.model_validate(
            {
                **config.model_dump(mode="json"),
                "output": {"directory": str(root / "evidence" / label)},
            }
        )
        result = run_decision_analysis(config)
        repeated = run_decision_analysis(config)
        assert result.model_dump() == repeated.model_dump(), "scientific reproduction differs"
        for run in (result, repeated):
            verification = verify(run.artifact_directory, deep=True)
            assert verification.passed, verification
        report = generate_report(
            result.artifact_directory, root / "reports" / f"{label}.html", deep=True
        )
        generate_report(
            result.artifact_directory,
            root / "reports" / f"{label}-reproduction.html",
            compare=repeated.artifact_directory,
            deep=True,
        )
        audit[label] = {
            "fingerprint": result.fingerprint,
            "report": str(report),
            "artifact": str(result.artifact_directory),
            "reproduction": str(repeated.artifact_directory),
            "scientific_results_equal": True,
        }
        print(format_decision_summary(result))
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
