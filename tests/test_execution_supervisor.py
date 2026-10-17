"""Externally observable termination, crash isolation, and scientific equivalence."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from time import monotonic, sleep

import pytest

from benchforge.artifacts.verify import verify
from benchforge.catalog import Catalog
from benchforge.core.config import OutputConfig, load_benchmark_config
from benchforge.execution import ExecutionOptions, run_benchmark_suite
from benchforge.execution.supervisor import supervise
from tests.process_workers import job

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX supervision contracts")


def _dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # Linux orphan zombies have stopped consuming resources, but cannot be
    # reaped by this coordinator (their parent is the system's init process).
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split()[2] == "Z"


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mode", ["block", "ignore", "descendant"])
def test_timeout_actually_terminates_worker_and_descendants(
    tmp_path: Path, workers: int, mode: str
) -> None:
    start = monotonic()
    outcomes = list(
        supervise(
            job,
            [(mode, str(tmp_path)), ("success", str(tmp_path))],
            workers=workers,
            options=ExecutionOptions(candidate_timeout=3),
            deadline=None,
        )
    )
    assert monotonic() - start < 9
    assert {o.index: o.reason for o in outcomes} == {0: "CandidateTimeout", 1: None}
    assert _dead(int((tmp_path / f"{mode}.pid").read_text()))
    if mode == "descendant":
        assert _dead(int((tmp_path / "descendant.child").read_text()))


def test_crash_reclaims_capacity_and_keeps_other_results(tmp_path: Path) -> None:
    outcomes = list(
        supervise(
            job,
            [(mode, str(tmp_path)) for mode in ["crash", "success", "fast"]],
            workers=1,
            options=ExecutionOptions(candidate_timeout=5),
            deadline=None,
        )
    )
    assert [o.reason for o in outcomes] == ["WorkerCrash", None, None]
    assert [o.result for o in outcomes] == [None, "success", "fast"]
    assert _dead(int((tmp_path / "crash.pid").read_text()))


def test_completion_inversion_and_no_hidden_queue(tmp_path: Path) -> None:
    iterator = supervise(
        job,
        [(mode, str(tmp_path)) for mode in ["slow", "fast", "success"]],
        workers=2,
        options=ExecutionOptions(candidate_timeout=15),
        deadline=None,
    )
    with closing(iterator):
        first = next(iterator)
        assert first.index == 1
        # Third task cannot start while the coordinator is committing this result.
        assert not (tmp_path / "success.pid").exists()
        (tmp_path / "release").touch()
        remaining = list(iterator)
    assert sorted(o.index for o in [first, *remaining]) == [0, 1, 2]


def test_deadline_distinguishes_running_and_not_started(tmp_path: Path) -> None:
    outcomes = list(
        supervise(
            job,
            [("block", str(tmp_path)), ("success", str(tmp_path))],
            workers=1,
            options=ExecutionOptions(run_timeout=3),
            deadline=monotonic() + 3,
        )
    )
    assert [o.reason for o in outcomes] == ["RunTimeout", "NotStarted"]
    assert not (tmp_path / "success.pid").exists()
    assert _dead(int((tmp_path / "block.pid").read_text()))


def test_deadline_before_scheduling_and_empty_queue(tmp_path: Path) -> None:
    assert list(supervise(job, [], workers=1, options=ExecutionOptions(), deadline=None)) == []
    outcomes = list(
        supervise(
            job,
            [("success", str(tmp_path))],
            workers=1,
            options=ExecutionOptions(),
            deadline=monotonic() - 1,
        )
    )
    assert outcomes[0].reason == "NotStarted"
    assert not (tmp_path / "success.pid").exists()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("field", ["candidate_timeout", "run_timeout"])
def test_invalid_budgets(value: float, field: str) -> None:
    with pytest.raises(ValueError, match="finite and greater"):
        ExecutionOptions(**{field: value})


@pytest.mark.parametrize("example", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_generous_budget_preserves_full_evidence_and_catalog(tmp_path: Path, example: str) -> None:
    config = load_benchmark_config(f"configs/examples/{example}_suite.yaml")
    config = config.model_copy(
        update={
            "models": config.models[:2],
            "split": config.split.model_copy(update={"n_splits": 3}),
            "output": OutputConfig(directory=tmp_path / "published"),
        }
    )
    legacy = run_benchmark_suite(config, persist=False)
    bounded = run_benchmark_suite(
        config,
        execution_options=ExecutionOptions(30, 90),
        checkpoint_dir=tmp_path / "work",
    )
    assert not bounded.failures
    assert legacy.fingerprint == bounded.fingerprint
    assert [c.run_result for c in legacy.candidates] == [c.run_result for c in bounded.candidates]
    assert [(r.rank, r.model_identifier, r.metrics) for r in legacy.leaderboard] == [
        (r.rank, r.model_identifier, r.metrics) for r in bounded.leaderboard
    ]
    assert bounded.artifact_directory is not None
    assert verify(bounded.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        assert len(catalog.ingest(bounded.artifact_directory)) == 1
    reused = run_benchmark_suite(
        config,
        execution_options=ExecutionOptions(0.001, 0.001),
        checkpoint_dir=tmp_path / "work",
        resume=True,
        persist=False,
    )
    assert len(reused.checkpoint_reused) == 2
    assert reused.candidates == bounded.candidates


def test_exhausted_run_retries_without_success_checkpoints(tmp_path: Path) -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    config = config.model_copy(update={"models": config.models[:1]})
    stopped = run_benchmark_suite(
        config,
        execution_options=ExecutionOptions(run_timeout=0.0001),
        checkpoint_dir=tmp_path / "work",
        persist=False,
    )
    assert not stopped.candidates
    assert stopped.failures[0].error_type == "NotStarted"
    assert list((tmp_path / "work" / "tasks").glob("*.json")) == []
    resumed = run_benchmark_suite(
        config,
        checkpoint_dir=tmp_path / "work",
        resume=True,
        execution_options=ExecutionOptions(30),
        persist=False,
    )
    assert len(resumed.candidates) == 1


def test_real_sigint_cleans_workers_releases_lock_and_resumes(tmp_path: Path) -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    config = config.model_copy(
        update={
            "models": config.models[:2],
            "output": OutputConfig(directory=tmp_path / "published"),
        }
    )
    source = tmp_path / "config.json"
    source.write_text(config.model_dump_json())
    script = """
import signal, sys
from pathlib import Path
from benchforge.core.config import BenchmarkConfig
from benchforge.execution import ExecutionOptions, run_benchmark_suite
from benchforge.execution.checkpoints import CheckpointWorkspace
from benchforge.execution import supervisor
popen = supervisor.subprocess.Popen

def tracked(*args, **kwargs):
    worker = popen(*args, **kwargs)
    with Path(sys.argv[3] + ".pids").open("a") as handle:
        handle.write(str(worker.pid) + "\\n")
    return worker

supervisor.subprocess.Popen = tracked
original = CheckpointWorkspace.save

def interrupt(self, candidate):
    original(self, candidate)
    Path(sys.argv[3]).touch()
    signal.pause()

CheckpointWorkspace.save = interrupt
run_benchmark_suite(BenchmarkConfig.model_validate_json(Path(sys.argv[1]).read_text()),
    workers=2, checkpoint_dir=Path(sys.argv[2]), execution_options=ExecutionOptions(30))
"""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            str(tmp_path / "work"),
            str(tmp_path / "ready"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        stop = monotonic() + 20
        while not (tmp_path / "ready").exists() and monotonic() < stop:
            assert child.poll() is None
            sleep(0.02)
        assert (tmp_path / "ready").exists()
        child.send_signal(signal.SIGINT)
        child.wait(timeout=5)
        assert child.returncode != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    for pid in (tmp_path / "ready.pids").read_text().splitlines():
        assert _dead(int(pid))
    assert not (tmp_path / "published").exists()
    assert len(list((tmp_path / "work" / "tasks").glob("*.json"))) == 1
    diagnostic = json.loads((tmp_path / "work" / "execution.json").read_text())
    assert any(row["outcome"] == "Cancelled" for row in diagnostic["candidates"])
    resumed = run_benchmark_suite(
        config,
        checkpoint_dir=tmp_path / "work",
        resume=True,
        execution_options=ExecutionOptions(30),
    )
    assert len(resumed.checkpoint_reused) == 1
    assert len(resumed.checkpoint_executed) == 1
    assert not resumed.failures
    assert resumed.artifact_directory is not None
    assert verify(resumed.artifact_directory, deep=True).passed


def test_startup_and_coordinator_errors_propagate_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchforge.execution import supervisor

    original = supervisor.subprocess.Popen
    launched = []

    def fail_second(*args: object, **kwargs: object) -> object:
        if launched:
            raise OSError("startup refused")
        worker = original(*args, **kwargs)
        launched.append(worker)
        return worker

    monkeypatch.setattr(supervisor.subprocess, "Popen", fail_second)
    with pytest.raises(OSError, match="startup refused"):
        list(
            supervise(
                job,
                [("block", str(tmp_path)), ("fast", str(tmp_path))],
                workers=2,
                options=ExecutionOptions(5),
                deadline=None,
            )
        )
    assert launched[0].poll() is not None
    monkeypatch.setattr(supervisor.subprocess, "Popen", original)

    def broken_callback(index: int) -> None:
        raise LookupError("coordinator defect")

    with pytest.raises(LookupError, match="coordinator defect"):
        list(
            supervise(
                job,
                [("block", str(tmp_path))],
                workers=1,
                options=ExecutionOptions(5),
                deadline=None,
                on_start=broken_callback,
            )
        )


def test_cleanup_failure_is_visible_and_worker_is_killed(tmp_path: Path) -> None:
    from benchforge.execution.supervisor import _Active, _cleanup

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    actual_wait = process.wait

    def failed_wait(timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(process.args, timeout or 0)

    try:
        process.wait = failed_wait
        with pytest.raises(RuntimeError, match="confirm worker exit"):
            _cleanup(_Active(process, monotonic(), tmp_path / "unused"))
    finally:
        process.wait = actual_wait
        process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("identifier", [".", ".."])
def test_path_component_identifiers_rejected_before_execution(identifier: str) -> None:
    from benchforge.core.config import ModelConfig

    with pytest.raises(ValueError, match="dot path"):
        ModelConfig(name="dummy_classifier", id=identifier)


def test_no_successes_publish_explicit_unranked_verifiable_evidence(tmp_path: Path) -> None:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    config = config.model_copy(update={"output": OutputConfig(directory=tmp_path / "published")})
    result = run_benchmark_suite(config, execution_options=ExecutionOptions(run_timeout=0.0001))
    assert len(result.failures) == len(config.models)
    assert all(row.rank is None and not row.metrics for row in result.leaderboard)
    assert result.artifact_directory is not None
    assert verify(result.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        assert len(catalog.ingest(result.artifact_directory)) == 1


def test_worker_limit_is_observed_at_every_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchforge.execution import supervisor

    original = supervisor.subprocess.Popen
    processes = []
    observed = []

    def tracked(*args: object, **kwargs: object) -> object:
        active = sum(process.poll() is None for process in processes)
        assert active < 2
        observed.append(active + 1)
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", tracked)
    results = list(
        supervise(
            job,
            [("success", str(tmp_path))] * 5,
            workers=2,
            options=ExecutionOptions(10),
            deadline=None,
        )
    )
    assert len(results) == 5
    assert max(observed) == 2
    assert all(process.poll() is not None for process in processes)


def test_partial_timeout_checkpoints_only_success_and_resume_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchforge.core.config import ModelConfig
    from benchforge.execution import benchmark
    from tests.process_workers import candidate_with_block

    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    config = config.model_copy(
        update={
            "models": (
                ModelConfig(name="dummy_classifier", id="success"),
                ModelConfig(name="dummy_classifier", id="blocked"),
            )
        }
    )
    original = benchmark._candidate_process_entry
    monkeypatch.setattr(benchmark, "_candidate_process_entry", candidate_with_block)
    stopped = run_benchmark_suite(
        config,
        checkpoint_dir=tmp_path / "work",
        execution_options=ExecutionOptions(3),
        persist=False,
    )
    assert [c.model_identifier for c in stopped.candidates] == ["success"]
    assert [f.error_type for f in stopped.failures] == ["CandidateTimeout"]
    assert len(list((tmp_path / "work" / "tasks").glob("*.json"))) == 1
    monkeypatch.setattr(benchmark, "_candidate_process_entry", original)
    resumed = run_benchmark_suite(
        config,
        checkpoint_dir=tmp_path / "work",
        resume=True,
        execution_options=ExecutionOptions(30),
        persist=False,
    )
    assert resumed.checkpoint_reused == ("success",)
    assert resumed.checkpoint_executed == ("blocked",)
    assert not resumed.failures


def test_interruption_during_publication_preserves_checkpoints_without_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchforge.storage import suite

    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    config = config.model_copy(
        update={
            "models": config.models[:1],
            "output": OutputConfig(directory=tmp_path / "published"),
        }
    )

    def interrupted_seal(path: Path) -> None:
        raise KeyboardInterrupt("interrupted before sealing")

    original = suite.seal
    monkeypatch.setattr(suite, "seal", interrupted_seal)
    with pytest.raises(KeyboardInterrupt, match="before sealing"):
        run_benchmark_suite(
            config,
            checkpoint_dir=tmp_path / "work",
            execution_options=ExecutionOptions(30),
        )
    partial = next((tmp_path / "published").iterdir())
    assert not (partial / "manifest.json").exists()
    assert not verify(partial).passed
    monkeypatch.setattr(suite, "seal", original)
    resumed = run_benchmark_suite(
        config,
        checkpoint_dir=tmp_path / "work",
        resume=True,
        execution_options=ExecutionOptions(30),
    )
    assert len(resumed.checkpoint_reused) == 1
    assert resumed.artifact_directory is not None
    assert verify(resumed.artifact_directory, deep=True).passed
