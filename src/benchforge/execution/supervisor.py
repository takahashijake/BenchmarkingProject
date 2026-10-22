"""Bounded subprocess scheduling; no fitting or scientific policy here."""

from __future__ import annotations

import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import cast


@dataclass(frozen=True)
class ExecutionOptions:
    candidate_timeout: float | None = None
    run_timeout: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_timeout", self.candidate_timeout),
            ("run_timeout", self.run_timeout),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.bounded and os.name != "posix":
            raise ValueError("hard execution budgets require POSIX process groups")

    @property
    def bounded(self) -> bool:
        return self.candidate_timeout is not None or self.run_timeout is not None


@dataclass(frozen=True)
class WorkerOutcome[R]:
    index: int
    result: R | None
    reason: str | None
    elapsed: float
    exit_code: int | None = None


@dataclass
class _Active:
    process: subprocess.Popen[bytes]
    started: float
    output: Path


def _cleanup(worker: _Active) -> None:
    """Reap the direct worker; POSIX also kills descendants in its session group."""
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(worker.process.pid, signal.SIGKILL)
    elif worker.process.poll() is None:
        worker.process.kill()
    try:
        worker.process.wait(timeout=2)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("candidate cleanup could not confirm worker exit") from exc


def supervise[T, R](
    function: Callable[[T], R],
    tasks: Sequence[T],
    *,
    workers: int,
    options: ExecutionOptions,
    deadline: float | None,
    on_start: Callable[[int], None] | None = None,
) -> Generator[WorkerOutcome[R], None, None]:
    """Yield completed work after cleanup; caller owns checkpoint commits.

    Pickles are ephemeral IPC in a private directory, never recovery inputs.
    A result is read only after exit, preventing partial reads/pipe deadlocks.
    Startup, imports and result serialization count toward candidate wall time.
    Closing this iterator also terminates all outstanding workers.
    """
    if os.name != "posix" and (options.bounded or deadline is not None):
        raise ValueError("hard execution budgets require POSIX process groups")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be at least 1")
    active: dict[int, _Active] = {}
    next_index = 0
    with tempfile.TemporaryDirectory(prefix="benchforge-ipc-") as temporary:
        root = Path(temporary)
        try:
            while active or next_index < len(tasks):
                now = monotonic()
                exhausted = deadline is not None and now >= deadline
                # Never build a hidden queue: serialize only when capacity is available.
                while not exhausted and next_index < len(tasks) and len(active) < workers:
                    index = next_index
                    started = monotonic()
                    source = root / f"{index}.input"
                    output = root / f"{index}.output"
                    with source.open("wb") as handle:
                        pickle.dump((function, tasks[index]), handle)
                    if deadline is not None and monotonic() >= deadline:
                        source.unlink()
                        exhausted = True
                        break
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "benchforge.execution.worker",
                            str(source),
                            str(output),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=os.name == "posix",
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "BLIS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1",
                        },
                    )
                    active[index] = _Active(process, started, output)
                    next_index += 1
                    if on_start is not None:
                        on_start(index)
                    exhausted = deadline is not None and monotonic() >= deadline
                for index, worker in list(active.items()):
                    elapsed = monotonic() - worker.started
                    # Deadline has priority over accepting a late result.
                    run_expired = deadline is not None and monotonic() >= deadline
                    candidate_expired = (
                        options.candidate_timeout is not None
                        and elapsed >= options.candidate_timeout
                    )
                    code = worker.process.poll()
                    if run_expired or candidate_expired or code is not None:
                        _cleanup(worker)
                        del active[index]
                        reason = (
                            "RunTimeout"
                            if run_expired
                            else "CandidateTimeout"
                            if candidate_expired
                            else "WorkerCrash"
                            if code != 0
                            else None
                        )
                        result: R | None = None
                        if reason is None:
                            # Invalid IPC is an orchestration defect, not estimator failure.
                            with worker.output.open("rb") as handle:
                                result = cast(R, pickle.load(handle))
                        (root / f"{index}.input").unlink()
                        worker.output.unlink(missing_ok=True)
                        yield WorkerOutcome(index, result, reason, elapsed, code)
                if deadline is not None and monotonic() >= deadline:
                    for index in range(next_index, len(tasks)):
                        yield WorkerOutcome(index, None, "NotStarted", 0.0)
                    next_index = len(tasks)
                if active:
                    sleep(0.01)
        finally:
            errors: list[Exception] = []
            for worker in active.values():
                try:
                    _cleanup(worker)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise ExceptionGroup("candidate cleanup failed", errors)
