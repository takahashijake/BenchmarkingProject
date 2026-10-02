from pathlib import Path

import pytest

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
