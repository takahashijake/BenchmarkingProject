"""Scientific equivalence, public policy and durable bounded search execution."""

from __future__ import annotations

import dataclasses
import json
import os
from contextlib import closing, suppress

import numpy as np
import pytest

from benchforge.artifacts.verify import verify
from benchforge.automl.runner import run_automl
from benchforge.catalog import Catalog
from benchforge.cli import main
from benchforge.core.config import SearchModelConfig, load_automl_config, load_search_config
from benchforge.data.identity import dataset_digest
from benchforge.data.registry import default_dataset_registry
from benchforge.search import parallel, runner
from benchforge.search.recovery import CheckpointPublicationError, SearchWorkspace
from tests.test_search_recovery import automl_config, config_for


def science(value):
    """Compare every scientific field, including folds, final study and trial failures."""
    if dataclasses.is_dataclass(value):
        return {
            f.name: science(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if "duration" not in f.name
            and not f.name.startswith("checkpoint_")
            and not f.name.startswith("execution_")
            and f.name != "artifact_directory"
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: science(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [science(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@pytest.mark.parametrize("example", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
@pytest.mark.parametrize("final", [False, True])
def test_portable_complete_scientific_equivalence(tmp_path, example, final):
    config = config_for(tmp_path, example)
    regression = example == "diabetes_regression"
    tree = "random_forest_regressor" if regression else "random_forest_classifier"
    config = config.model_copy(
        update={
            "models": (
                *config.models,
                SearchModelConfig(name=tree, id="tree", parameters={"n_estimators": 8}),
            ),
            "final_search": config.final_search.model_copy(update={"enabled": final}),
        }
    )
    reference = runner.run_search(config, persist=False)
    result = runner.run_search(config, workers=2)
    assert science(reference) == science(result)
    assert result.execution_mode == "subprocess"
    assert result.execution_planned_tasks == 6 + int(final)
    assert len(result.execution_diagnostics) == 6 + int(final)
    assert verify(result.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        assert catalog.ingest(result.artifact_directory)


@pytest.mark.parametrize("example", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_automl_science_and_recovery(tmp_path, example, monkeypatch):
    original = load_automl_config(f"configs/examples/{example}_automl.yaml")
    tuned = "ridge_regressor" if example == "diabetes_regression" else "logistic_regression"
    fixed = "dummy_regressor" if example == "diabetes_regression" else "dummy_classifier"
    config = original.model_copy(
        update={
            "models": original.models.model_copy(update={"include": (tuned, fixed)}),
            "outer_split": original.outer_split.model_copy(update={"n_splits": 2}),
            "inner_split": original.inner_split.model_copy(update={"n_splits": 2}),
            "budget": original.budget.model_copy(
                update={"total_trials": 6, "final_search_trials": 2}
            ),
            "output": automl_config(tmp_path).output,
        }
    )
    reference = run_automl(config, persist=False)
    result = run_automl(config, workers=2, checkpoint_dir=tmp_path / "work")
    assert science(reference) == science(result)
    assert verify(result.artifact_directory, deep=True).passed

    def forbidden(*args, **kwargs):
        raise AssertionError("complete recovery must not launch or fit")

    monkeypatch.setattr(parallel, "supervise", forbidden)
    monkeypatch.setattr(runner, "_final_search", forbidden)
    # Changing to serial requires no subprocesses either.
    resumed = run_automl(config, workers=1, checkpoint_dir=tmp_path / "work", resume=True)
    assert len(resumed.search_result.checkpoint_reused) == 5
    assert science(resumed) == science(reference)


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "2", None, 10**9])
@pytest.mark.parametrize("automl", [False, True])
def test_bad_worker_counts_fail_before_activation(tmp_path, monkeypatch, workers, automl):
    from benchforge.providers import runtime

    def forbidden(*args):
        raise AssertionError("must validate before activation")

    monkeypatch.setattr(runtime, "activate", forbidden)
    with pytest.raises(ValueError, match="workers"):
        (run_automl if automl else runner.run_search)(
            automl_config(tmp_path) if automl else config_for(tmp_path),
            workers=workers,
        )


@pytest.mark.parametrize("automl", [False, True])
def test_providers_rejected_before_import(tmp_path, monkeypatch, automl):
    from benchforge.providers import runtime

    config = (
        automl_config(tmp_path).model_copy(
            update={"providers": load_search_config("configs/providers/search.yaml").providers}
        )
        if automl
        else load_search_config("configs/providers/search.yaml")
    )

    def forbidden(*args):
        raise AssertionError("provider must not activate")

    monkeypatch.setattr(runtime, "activate", forbidden)
    with pytest.raises(ValueError, match="providers require serial"):
        (run_automl if automl else runner.run_search)(config, workers=2)


@pytest.mark.parametrize("field", ["outer", "final", "jobs", "registry"])
def test_incompatible_policy_fails_before_dispatch(tmp_path, monkeypatch, field):
    config = config_for(tmp_path)
    kwargs = {}
    if field == "outer":
        config = config.model_copy(
            update={
                "budget": config.budget.model_copy(update={"timeout_seconds_per_outer_fold": 5})
            }
        )
    elif field == "final":
        config = config.model_copy(
            update={"final_search": config.final_search.model_copy(update={"timeout_seconds": 5})}
        )
    elif field == "jobs":
        config = config.model_copy(
            update={
                "models": (
                    SearchModelConfig(name="random_forest_classifier", parameters={"n_jobs": -1}),
                )
            }
        )
    else:
        from benchforge.models.registry import ModelRegistry

        kwargs["model_registry"] = ModelRegistry({})
    with pytest.raises(ValueError, match="timeouts|n_jobs=1|built-in registries"):
        runner.run_search(config, workers=2, checkpoint_dir=tmp_path / "work", **kwargs)
    assert not (tmp_path / "work").exists()


def test_serial_default_never_dispatches(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default is direct serial execution")

    monkeypatch.setattr(parallel, "supervise", forbidden)
    result = runner.run_search(config_for(tmp_path), persist=False)
    assert result.execution_workers == 1
    assert result.execution_mode == "serial"


def test_portable_full_recovery_skips_dispatch_and_corruption_recomputes(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    fresh = runner.run_search(config, workers=2, checkpoint_dir=work, persist=False)
    original = parallel.supervise
    launched = []

    def tracked(function, tasks, **kwargs):
        launched.extend(t.task.task_id for t in tasks)
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(parallel, "supervise", tracked)
    resumed = runner.run_search(config, workers=2, checkpoint_dir=work, resume=True, persist=False)
    assert launched == []
    assert len(resumed.checkpoint_reused) == 5
    assert not resumed.checkpoint_executed
    path = next(
        p
        for p in (work / "tasks").glob("*.json")
        if json.loads(p.read_text())["payload"].get("trials")
    )
    path.write_text("corrupt")
    changed = runner.run_search(config, workers=2, checkpoint_dir=work, resume=True)
    assert len(launched) == 1
    assert len(changed.checkpoint_invalid) == 1
    assert len(changed.checkpoint_reused) == 4
    assert science(changed) == science(fresh)
    assert verify(changed.artifact_directory, deep=True).passed


def test_parallel_workers_audit_leakage_preprocessing_and_private_state(tmp_path, monkeypatch):
    from tests.parallel_search_workers import audited

    config = config_for(tmp_path).model_copy(
        update={
            "final_search": config_for(tmp_path).final_search.model_copy(update={"enabled": False})
        }
    )
    dataset = default_dataset_registry.resolve(config.dataset)
    before = dataset_digest(dataset)
    reference = runner.run_search(config, dataset=dataset, persist=False)
    monkeypatch.setattr(parallel, "_outer_process_entry", audited)
    result = runner.run_search(config, dataset=dataset, workers=2, persist=False)
    assert science(reference) == science(result)
    assert dataset_digest(dataset) == before


def test_portable_crash_isolated_and_successful_folds_recover(tmp_path, monkeypatch):
    from tests.parallel_search_workers import crash_first

    config = config_for(tmp_path)
    work = tmp_path / "work"
    monkeypatch.setattr(parallel, "_outer_process_entry", crash_first)
    partial = runner.run_search(config, workers=2, checkpoint_dir=work)
    assert [f.error_type for f in partial.failures] == ["WorkerCrash"]
    assert partial.failures[0].model_identifier == "tuned"
    assert {f.model_identifier for f in partial.families} == {"fixed"}
    assert partial.leaderboard[-1].rank is None
    from benchforge.analysis.summary import format_automl_summary, format_search_summary
    from benchforge.automl.planner import build_automl_plan
    from benchforge.automl.results import AutoMLResult

    summary = format_search_summary(partial)
    assert "nested search incomplete" in summary
    assert "WorkerCrash" in summary
    assert "Ranks compare complete families only" in summary
    auto = AutoMLResult(
        fingerprint="format-only",
        plan=build_automl_plan(automl_config(tmp_path)),
        search_result=partial,
        total_duration_seconds=0.0,
    )
    assert "AutoML incomplete" in format_automl_summary(auto)
    assert "WorkerCrash" in format_automl_summary(auto)
    assert len(list((work / "tasks").glob("*.json"))) == 4  # 3 outer + fixed final
    assert verify(partial.artifact_directory, deep=True).passed
    monkeypatch.undo()
    resumed = runner.run_search(config, workers=2, checkpoint_dir=work, resume=True)
    reference = runner.run_search(config, persist=False)
    assert "outer:tuned:1" in resumed.checkpoint_reused
    assert resumed.checkpoint_executed == ("outer:tuned:0", "final:tuned")
    assert science(reference) == science(resumed)


def test_trial_failures_and_all_failed_are_explicit(tmp_path):
    config = config_for(tmp_path)
    failed = SearchModelConfig(name="logistic_regression", id="bad", parameters={"C": -1.0})
    config = config.model_copy(update={"models": (failed, config.models[1])})
    result = runner.run_search(config, workers=2, checkpoint_dir=tmp_path / "work", persist=False)
    assert result.failures[0].error_type == "FamilySearchError"
    assert len(result.failures[0].trials) == 4
    assert all(t.state == "FAIL" and t.error_type for t in result.failures[0].trials)
    only = config.model_copy(update={"models": (failed,)})
    with pytest.raises(runner.FamilySearchError, match="every search family failed"):
        runner.run_search(only, workers=2, persist=False)


@pytest.mark.parametrize("fixture", ["bad_ipc", "corrupt_result", "unexpected"])
def test_ipc_and_unexpected_worker_defects_abort(tmp_path, monkeypatch, fixture):
    from tests import parallel_search_workers

    monkeypatch.setattr(parallel, "_outer_process_entry", getattr(parallel_search_workers, fixture))
    with pytest.raises((RuntimeError, ValueError), match="IPC|seed mismatch|unexpected worker"):
        runner.run_search(config_for(tmp_path), workers=2, persist=False)


def test_publication_errors_abort_and_release_lock(tmp_path, monkeypatch):
    original = SearchWorkspace.save_outer

    def broken(*args):
        raise CheckpointPublicationError("publication failed")

    monkeypatch.setattr(SearchWorkspace, "save_outer", broken)
    with pytest.raises(CheckpointPublicationError, match="publication failed"):
        runner.run_search(config_for(tmp_path), workers=2, checkpoint_dir=tmp_path / "work")
    monkeypatch.setattr(SearchWorkspace, "save_outer", original)
    result = runner.run_search(
        config_for(tmp_path),
        workers=2,
        checkpoint_dir=tmp_path / "work",
        resume=True,
        persist=False,
    )
    assert not result.failures


def test_cli_workers_and_plan_only(tmp_path, monkeypatch, capsys):
    config = config_for(tmp_path)
    path = tmp_path / "search.json"
    path.write_text(config.model_dump_json())
    assert main(["search", str(path), "--workers", "2", "--workspace", str(tmp_path / "work")]) == 0
    assert "Execution mode: subprocess (2 workers)" in capsys.readouterr().out
    assert main(["search", str(path), "--workers", "1", "--resume", str(tmp_path / "work")]) == 0
    assert "Reused tasks: 5" in capsys.readouterr().out

    def forbidden(*args, **kwargs):
        raise AssertionError("planning never dispatches")

    monkeypatch.setattr(parallel, "supervise", forbidden)
    assert (
        main(
            [
                "automl",
                "configs/examples/breast_cancer_automl.yaml",
                "--workers",
                "2",
                "--plan-only",
            ]
        )
        == 0
    )
    assert "Execution mode:" not in capsys.readouterr().out


def _coordinator(tmp_path, monkeypatch, *, stage="normal", automl=False):
    """Coordinator process with independently controlled real fitting children."""
    import subprocess
    import sys
    from multiprocessing.connection import Listener

    listener = Listener(("127.0.0.1", 0), authkey=b"benchforge-test")
    # Only this test listener uses the implementation socket to bound accept.
    listener._listener._socket.settimeout(40)
    host, port = listener.address
    monkeypatch.setenv("BENCHFORGE_TEST_ADDRESS", f"{host}:{port}")
    config = automl_config(tmp_path) if automl else config_for(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    script = r"""
import sys
from pathlib import Path
from benchforge.core.config import SearchConfig, AutoMLConfig
from benchforge.search import parallel, runner
from benchforge.search.recovery import SearchWorkspace
from benchforge.automl.runner import run_automl
from tests.parallel_search_workers import controlled
parallel._outer_process_entry = controlled
path, work, stage, auto = sys.argv[1:]
cls = AutoMLConfig if auto == "True" else SearchConfig
config = cls.model_validate_json(Path(path).read_text())
save = SearchWorkspace.save_outer
commits = 0
def pause(label):
    print(label, flush=True)
    sys.stdin.readline()
def publish(self, candidate, fold, value):
    global commits
    save(self, candidate, fold, value)
    commits += 1
    print("committed:" + self.outer_key(candidate, fold), flush=True)
    if stage in {"after", "cancel"} and commits == 1:
        pause("boundary")
SearchWorkspace.save_outer = publish
if stage == "before":
    import benchforge.execution.checkpoint_io as io
    rename = io.os.replace
    def replace(src, dst):
        if Path(dst).parent.name == "tasks" and commits == 1:
            pause("boundary")
        return rename(src, dst)
    io.os.replace = replace
if stage == "sealing":
    import benchforge.storage.search as storage
    seal = storage.seal
    def interrupted(path):
        pause("boundary")
        return seal(path)
    storage.seal = interrupted
(run_automl if auto == "True" else runner.run_search)(config, workers=2, checkpoint_dir=Path(work))
print("done", flush=True)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(tmp_path / "work"), stage, str(automl)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return listener, child, config


def _line(child):
    import select
    from time import monotonic

    # Read exactly one line, avoiding TextIO buffering across select handshakes.
    stop = monotonic() + 40
    data = bytearray()
    while not data.endswith(b"\n"):
        assert select.select([child.stdout], [], [], max(0, stop - monotonic()))[0], (
            "coordinator boundary timed out"
        )
        byte = os.read(child.stdout.fileno(), 1)
        assert byte, "coordinator exited before boundary"
        data.extend(byte)
    return data.decode().strip()


def _stop(child):
    import signal
    import subprocess

    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=10)


def _receive(connection):
    assert connection.poll(40), "worker handshake timed out"
    return connection.recv()


def _finish(connection):
    connection.send("run")
    assert _receive(connection)[0] == "complete"
    connection.send("publish")


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker signal contract")
@pytest.mark.parametrize("phase", ["study", "refit"])
def test_kill_worker_during_science_preserves_other_tasks(tmp_path, monkeypatch, phase):
    import signal

    listener, child, config = _coordinator(tmp_path, monkeypatch)
    connections = []
    try:
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
        identified = {_receive(c)[0]: c for c in connections}
        first, blocked = identified["outer:tuned:0"], identified["outer:tuned:1"]
        blocked.send(phase)
        _, pid = _receive(blocked)
        _finish(first)
        assert _line(child) == "committed:outer:tuned:0"
        os.kill(pid, signal.SIGKILL)
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
            assert _receive(connection)[0].startswith("outer:fixed:")
            _finish(connection)
        stdout, stderr = child.communicate(timeout=40)
        assert child.returncode == 0, stderr
        assert "done" in stdout
    finally:
        _stop(child)
        for connection in connections:
            connection.close()
        listener.close()
    work = tmp_path / "work"
    assert len(list((work / "tasks").glob("*.json"))) == 4
    recovered = runner.run_search(config, workers=2, checkpoint_dir=work, resume=True)
    assert "outer:tuned:0" in recovered.checkpoint_reused
    assert recovered.checkpoint_executed == ("outer:tuned:1", "final:tuned")
    assert science(recovered) == science(runner.run_search(config, persist=False))


@pytest.mark.skipif(os.name != "posix", reason="POSIX interruption and lock contract")
@pytest.mark.parametrize("stage", ["before", "after", "cancel", "sealing"])
@pytest.mark.parametrize("automl", [False, True])
def test_real_interruption_atomic_publication_lock_and_resume(tmp_path, monkeypatch, stage, automl):
    import signal

    from tests.test_execution_supervisor import _dead

    listener, child, config = _coordinator(tmp_path, monkeypatch, stage=stage, automl=automl)
    connections = []
    pids = []
    try:
        initial = []
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
            key, pid = _receive(connection)
            pids.append(pid)
            initial.append((key, connection))
        initial.sort(key=lambda pair: pair[0])
        _finish(initial[0][1])
        assert _line(child).startswith("committed:")
        if stage in {"before", "sealing"}:
            _finish(initial[1][1])
        if stage == "sealing":
            assert _line(child).startswith("committed:")
            for _ in range(2):
                connection = listener.accept()
                connections.append(connection)
                _, pid = _receive(connection)
                pids.append(pid)
                _finish(connection)
                assert _line(child).startswith("committed:")
        assert _line(child) == "boundary"
        work = tmp_path / "work"
        # An independent process owns the lock, including while publishing/sealing.
        with pytest.raises(ValueError, match="already in use"):
            (run_automl if automl else runner.run_search)(
                config, workers=2, checkpoint_dir=work, resume=True, persist=False
            )
        committed = len(list((work / "tasks").glob("*.json")))
        assert committed == (5 if stage == "sealing" else 1)
        # Cooperative interruption must close/reap every active worker before unlock.
        child.send_signal(signal.SIGINT)
        _, stderr = child.communicate(timeout=15)
        assert child.returncode != 0
        assert all(_dead(pid) for pid in pids), stderr
    finally:
        _stop(child)
        for connection in connections:
            connection.close()
        listener.close()
    run = run_automl if automl else runner.run_search
    reference = run(config, persist=False)
    resumed = run(config, workers=1, checkpoint_dir=work, resume=True)
    result = resumed.search_result if automl else resumed
    assert len(result.checkpoint_reused) == committed
    assert science(resumed) == science(reference)
    assert verify(resumed.artifact_directory, deep=True).passed
    if stage == "sealing":
        artifacts = list(config.output.directory.iterdir())
        assert any(not (p / "manifest.json").exists() for p in artifacts)


@pytest.mark.skipif(os.name != "posix", reason="bounded POSIX pipe handshake")
def test_completion_order_and_active_limit_are_independent_of_science(tmp_path, monkeypatch):
    listener, child, config = _coordinator(tmp_path, monkeypatch)
    connections = []
    try:
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
        tasks = {_receive(c)[0]: c for c in connections}
        # Force fold 1 to commit before fold 0, then permit later candidates to run.
        _finish(tasks["outer:tuned:1"])
        assert _line(child) == "committed:outer:tuned:1"
        _finish(tasks["outer:tuned:0"])
        assert _line(child) == "committed:outer:tuned:0"
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
            _receive(connection)
            _finish(connection)
        _, stderr = child.communicate(timeout=40)
        assert child.returncode == 0, stderr
    finally:
        _stop(child)
        for connection in connections:
            connection.close()
        listener.close()
    resumed = runner.run_search(config, workers=2, checkpoint_dir=tmp_path / "work", resume=True)
    assert science(resumed) == science(runner.run_search(config, persist=False))


def test_portable_supervisor_uses_direct_process_cleanup_without_posix_groups(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from benchforge.execution import supervisor
    from benchforge.execution.supervisor import ExecutionOptions
    from tests.process_workers import job

    monkeypatch.setattr(supervisor, "os", SimpleNamespace(name="nt", environ=os.environ))
    outcomes = list(
        supervisor.supervise(
            job,
            [("success", str(tmp_path)), ("crash", str(tmp_path))],
            workers=2,
            options=ExecutionOptions(),
            deadline=None,
        )
    )
    assert sorted(o.reason or "success" for o in outcomes) == ["WorkerCrash", "success"]
    original = supervisor.subprocess.Popen
    processes = []

    def tracked(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", tracked)
    with closing(
        supervisor.supervise(
            job,
            [("block", str(tmp_path)), ("fast", str(tmp_path))],
            workers=2,
            options=ExecutionOptions(),
            deadline=None,
        )
    ) as pending:
        assert next(pending).result == "fast"
        assert any(p.poll() is None for p in processes)
    assert all(p.poll() is not None for p in processes)
    with pytest.raises(ValueError, match="POSIX"):
        ExecutionOptions(candidate_timeout=1)


def test_plan_is_deeply_immutable_and_has_checkpoint_task_names(tmp_path):
    from benchforge.core.seeds import derive_seed
    from benchforge.splits.stratified import build_folds, build_inner_folds

    config = config_for(tmp_path)
    dataset = default_dataset_registry.resolve(config.dataset)
    outer = build_folds(dataset.target, config.outer_split, config.seed, config.task)
    inner = {
        f.fold_id: build_inner_folds(
            dataset.target,
            f,
            config.inner_split,
            derive_seed(config.seed, "inner_folds", f.fold_id),
            config.task,
        )
        for f in outer
    }
    plan = parallel.SearchExecutionPlan.build(config, outer, inner)
    assert tuple(t.task_id for t in plan.tasks) == tuple(
        SearchWorkspace.outer_key(c, f) for c in config.models for f in outer
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.tasks[0].task_id = "mutated"
    with pytest.raises(TypeError):
        plan.tasks[0].outer.train[0] = 10
    with pytest.raises(ValueError):
        plan.tasks[0].outer.to_fold().train_indices[0] = 10


def test_portable_launch_limit_unique_tasks_and_thread_caps(tmp_path, monkeypatch):
    from benchforge.execution import supervisor

    config = config_for(tmp_path)
    processes, task_ids = [], []
    popen = supervisor.subprocess.Popen
    entry = parallel.supervise

    def tracked(*args, **kwargs):
        assert sum(p.poll() is None for p in processes) < 2
        assert all(
            kwargs["env"][name] == "1"
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        )
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def dispatch(function, tasks, **kwargs):
        task_ids.extend(t.task.task_id for t in tasks)
        return entry(function, tasks, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", tracked)
    monkeypatch.setattr(parallel, "supervise", dispatch)
    runner.run_search(config, workers=2, checkpoint_dir=tmp_path / "work", persist=False)
    assert len(processes) == 4
    assert len(set(task_ids)) == len(task_ids) == 4
    assert all(p.poll() is not None for p in processes)
    runner.run_search(
        config, workers=2, checkpoint_dir=tmp_path / "work", resume=True, persist=False
    )
    assert len(processes) == 4


@pytest.mark.skipif(os.name != "posix", reason="uncatchable POSIX coordinator death")
def test_sigkill_coordinator_retains_committed_checkpoint(tmp_path, monkeypatch):
    import signal

    listener, child, config = _coordinator(tmp_path, monkeypatch, stage="after")
    connections = []
    pids = []
    try:
        for _ in range(2):
            connection = listener.accept()
            connections.append(connection)
            _, pid = _receive(connection)
            pids.append(pid)
        _finish(connections[0])
        assert _line(child).startswith("committed:")
        assert _line(child) == "boundary"
        child.kill()
        child.communicate(timeout=10)
        assert child.returncode == -signal.SIGKILL
    finally:
        _stop(child)
        # Uncatchable coordinator death cannot run its cleanup. The test owns
        # and explicitly terminates these isolated worker groups.
        for pid in pids:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        for connection in connections:
            connection.close()
        listener.close()
    recovered = runner.run_search(config, workers=2, checkpoint_dir=tmp_path / "work", resume=True)
    assert len(recovered.checkpoint_reused) == 1
    assert science(recovered) == science(runner.run_search(config, persist=False))


def test_refit_failure_retains_completed_trial_evidence(tmp_path, monkeypatch):
    from tests.parallel_search_workers import refit_failure

    monkeypatch.setattr(parallel, "_outer_process_entry", refit_failure)
    result = runner.run_search(config_for(tmp_path), workers=2, persist=False)
    failure = result.failures[0]
    assert failure.error_type == "FoldExecutionError"
    assert failure.configured_trials == failure.completed_trials == 4
    assert len(failure.trials) == 4
    assert all(t.state == "COMPLETE" for t in failure.trials)
    assert [f.model_identifier for f in result.families] == ["fixed"]
