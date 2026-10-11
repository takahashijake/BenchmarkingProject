"""Importable adversarial workers, executed in real isolated subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep


def job(task: tuple[str, str]) -> str:
    mode, directory = task
    root = Path(directory)
    (root / f"{mode}.pid").write_text(str(os.getpid()))
    if mode == "crash":
        os._exit(73)
    if mode == "descendant":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (root / "descendant.child").write_text(str(child.pid))
    if mode == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode in {"block", "descendant", "ignore"}:
        sleep(60)
    if mode == "slow":
        stop = monotonic() + 10
        while not (root / "release").exists() and monotonic() < stop:
            sleep(0.01)
    return mode


def candidate_with_block(arguments: object) -> object:
    from benchforge.execution.benchmark import _candidate_process_entry

    model = arguments[1]
    if model.id == "blocked":
        sleep(60)
    return _candidate_process_entry(arguments)
