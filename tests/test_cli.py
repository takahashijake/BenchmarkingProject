from pathlib import Path

import pytest
import yaml

from benchforge.cli import main


def test_cli_runs_example(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("configs/examples/breast_cancer_logreg.yaml")
        .read_text(encoding="utf-8")
        .replace("directory: artifacts", f"directory: {tmp_path / 'artifacts'}"),
        encoding="utf-8",
    )
    assert main(["run", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "BenchForge run complete" in output
    assert "Fingerprint:" in output


def test_cli_reports_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("task: binary_classification\nmetrics: []\n", encoding="utf-8")
    assert main(["run", str(config_path)]) == 2
    error = capsys.readouterr().err
    assert "benchforge: error:" in error
    assert "at least one metric is required" in error


def test_cli_runs_mixed_dataset_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = Path("configs/examples/mixed_tabular_suite.yaml")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["output"]["directory"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert main(["benchmark", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "BenchForge benchmark complete" in output
    assert "Leaderboard (highest measured mean first):" in output
    assert "30 rows × 5 features" in output


def test_cli_reports_invalid_benchmark_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(
        Path("configs/examples/breast_cancer_suite.yaml").read_text(encoding="utf-8")
    )
    data["models"] = []
    config_path = tmp_path / "invalid-benchmark.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert main(["benchmark", str(config_path)]) == 2
    error = capsys.readouterr().err
    assert "benchforge: error:" in error
    assert "at least one model is required" in error
