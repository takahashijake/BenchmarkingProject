"""Recovery contracts: reuse, integrity, isolation, and published evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchforge.artifacts.verify import verify
from benchforge.catalog import Catalog
from benchforge.cli import main
from benchforge.core.config import (
    BenchmarkConfig,
    ModelConfig,
    OutputConfig,
    load_benchmark_config,
)
from benchforge.execution import benchmark as execution
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.checkpoints import CheckpointWorkspace


def _config(tmp_path: Path) -> BenchmarkConfig:
    original = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    return original.model_copy(update={
        "split": original.split.model_copy(update={"n_splits": 3}),
        "models": (
            ModelConfig(name="dummy_classifier", id="baseline"),
            ModelConfig(name="logistic_regression", id="logreg", parameters={
                "max_iter": 300, "solver": "liblinear"
            }),
        ),
        "output": OutputConfig(directory=tmp_path / "published"),
    })


def _semantic_results(result: object) -> object:
    assert isinstance(result, execution.BenchmarkResult)
    return (
        result.fingerprint,
        [(c.model_identifier, c.run_result.fold_results, c.run_result.predictions)
         for c in result.candidates],
        [(r.rank, r.model_identifier, r.primary_metric_mean) for r in result.leaderboard],
    )


def test_resume_reuses_all_candidates_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    fresh = run_benchmark_suite(config, checkpoint_dir=workspace)
    assert set(fresh.checkpoint_executed) == {"baseline", "logreg"}
    assert fresh.checkpoint_reused == ()
    assert fresh.artifact_directory is not None
    assert verify(fresh.artifact_directory).passed

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("a valid checkpoint must not refit")

    monkeypatch.setattr(execution, "_execute_candidate", forbidden)
    resumed = run_benchmark_suite(
        config, checkpoint_dir=workspace, resume=True, persist=False
    )
    assert set(resumed.checkpoint_reused) == {"baseline", "logreg"}
    assert resumed.checkpoint_executed == ()
    assert _semantic_results(resumed) == _semantic_results(fresh)


def test_interrupted_run_preserves_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    original = execution._execute_candidate
    calls: list[str] = []

    def interrupted(*args: object, **kwargs: object) -> object:
        model = args[1]
        assert isinstance(model, ModelConfig)
        calls.append(model.id or model.name)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated interruption before the second fit")
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_execute_candidate", interrupted)
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    assert len(list((workspace / "tasks").glob("*.json"))) == 1

    executed: list[str] = []

    def observed(*args: object, **kwargs: object) -> object:
        model = args[1]
        assert isinstance(model, ModelConfig)
        executed.append(model.id or model.name)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_execute_candidate", observed)
    result = run_benchmark_suite(
        config, checkpoint_dir=workspace, resume=True, persist=False
    )
    assert result.checkpoint_reused == ("baseline",)
    assert executed == ["logreg"]
    assert result.checkpoint_executed == ("logreg",)


@pytest.mark.parametrize("mutation", ["truncate", "change_digest", "unsupported_schema"])
def test_invalid_checkpoint_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    fresh = run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    task_path = next((workspace / "tasks").glob("*.json"))
    if mutation == "truncate":
        task_path.write_text("{", encoding="utf-8")
    else:
        data = json.loads(task_path.read_text(encoding="utf-8"))
        if mutation == "change_digest":
            data["sha256"] = "0" * 64
        else:
            data["schema"] = 999
        task_path.write_text(json.dumps(data), encoding="utf-8")

    original = execution._execute_candidate
    calls: list[str] = []

    def observed(*args: object, **kwargs: object) -> object:
        model = args[1]
        assert isinstance(model, ModelConfig)
        calls.append(model.id or model.name)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_execute_candidate", observed)
    resumed = run_benchmark_suite(
        config, checkpoint_dir=workspace, resume=True, persist=False
    )
    assert len(resumed.checkpoint_invalid) == 1
    assert len(resumed.checkpoint_reused) == 1
    assert len(calls) == 1
    assert _semantic_results(fresh) == _semantic_results(resumed)


def test_resume_rejects_changed_seed_or_models(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    with pytest.raises(ValueError, match="incompatible"):
        run_benchmark_suite(
            config.model_copy(update={"seed": config.seed + 1}),
            checkpoint_dir=workspace, resume=True, persist=False,
        )
    with pytest.raises(ValueError, match="incompatible"):
        run_benchmark_suite(
            config.model_copy(update={"models": config.models[:1]}),
            checkpoint_dir=workspace, resume=True, persist=False,
        )


def test_new_workspace_never_overwrites_existing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)


def test_workspace_lock_blocks_concurrent_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    dataset = execution.default_dataset_registry.resolve(config.dataset)
    folds = execution.build_folds(dataset.target, config.split, config.seed, config.task)
    with (
        CheckpointWorkspace(workspace, config, dataset, folds, resume=True),
        pytest.raises(ValueError, match="already in use"),
    ):
        run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)


@pytest.mark.parametrize("filename", ["untrusted.json", ".incomplete.tmp"])
def test_unsafe_checkpoint_symlink_is_rejected(tmp_path: Path, filename: str) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    extra = workspace / "tasks" / filename
    try:
        extra.symlink_to(workspace / "workspace.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="unexpected checkpoint"):
        run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)


def test_parallel_resume_has_same_scientific_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sequential = run_benchmark_suite(config, persist=False, workers=1)
    parallel = run_benchmark_suite(
        config, checkpoint_dir=tmp_path / "work", workers=2, persist=False
    )
    resumed = run_benchmark_suite(
        config, checkpoint_dir=tmp_path / "work", resume=True, workers=2, persist=False
    )
    assert _semantic_results(sequential) == _semantic_results(parallel)
    assert _semantic_results(parallel) == _semantic_results(resumed)


def test_real_process_exit_after_first_commit_is_resumable(tmp_path: Path) -> None:
    """OS process termination cannot turn completed JSON into partial success."""
    import subprocess
    import sys

    config = _config(tmp_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    workspace = tmp_path / "crash-workspace"
    script = """
import os
import sys
from pathlib import Path
from benchforge.core.config import BenchmarkConfig
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.checkpoints import CheckpointWorkspace

config = BenchmarkConfig.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
save = CheckpointWorkspace.save

def crash_after_committing(self, candidate):
    save(self, candidate)
    os._exit(73)

CheckpointWorkspace.save = crash_after_committing
run_benchmark_suite(config, checkpoint_dir=Path(sys.argv[2]), persist=False)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(workspace)],
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert child.returncode == 73, child.stderr
    assert len(list((workspace / "tasks").glob("*.json"))) == 1
    result = run_benchmark_suite(
        config, checkpoint_dir=workspace, resume=True, persist=False
    )
    assert result.checkpoint_reused == ("baseline",)
    assert result.checkpoint_executed == ("logreg",)
    assert not result.failures


def test_cli_workspace_then_resume_reports_reuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "benchmark.yaml"
    # JSON is a YAML subset; this keeps the exact typed config for both CLI runs.
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    workspace = tmp_path / "cli-work"
    assert main(["benchmark", str(config_path), "--workspace", str(workspace)]) == 0
    fresh = capsys.readouterr().out
    assert "Newly evaluated candidates: 2" in fresh
    assert "Reused candidates: 0" in fresh
    assert main(["benchmark", str(config_path), "--resume", str(workspace)]) == 0
    recovered = capsys.readouterr().out
    assert "Reused candidates: 2" in recovered
    assert "Newly evaluated candidates: 0" in recovered


@pytest.mark.parametrize(
    "example",
    [
        "configs/examples/iris_multiclass_suite.yaml",
        "configs/examples/diabetes_regression_suite.yaml",
    ],
)
def test_multiclass_and_regression_resume_preserve_results(
    tmp_path: Path, example: str
) -> None:
    original = load_benchmark_config(example)
    config = original.model_copy(update={
        "models": original.models[:2],
        "split": original.split.model_copy(update={"n_splits": 3}),
        "output": OutputConfig(directory=tmp_path / "published"),
    })
    sequential = run_benchmark_suite(config, persist=False)
    workspace = tmp_path / "recovery"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    resumed = run_benchmark_suite(
        config, checkpoint_dir=workspace, resume=True, persist=False
    )
    assert not resumed.failures
    assert len(resumed.checkpoint_reused) == 2
    assert _semantic_results(resumed) == _semantic_results(sequential)


def test_resumed_benchmark_publishes_catalog_compatible_sealed_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    resumed = run_benchmark_suite(config, checkpoint_dir=workspace, resume=True)
    assert resumed.artifact_directory is not None
    assert verify(resumed.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "index.sqlite3") as catalog:
        ids = catalog.ingest(resumed.artifact_directory)
        assert len(ids) == 1
        assert catalog.verify(ids[0], deep=True)[0].passed
