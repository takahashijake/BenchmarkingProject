from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from benchforge.analysis.summary import format_run_summary
from benchforge.core.config import load_run_config
from benchforge.execution.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchforge", description="Run reproducible classical-ML benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute a benchmark configuration")
    run_parser.add_argument("config", help="path to a YAML run configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config)
        result = run_benchmark(config)
    except (ValidationError, ValueError, RuntimeError) as exc:
        print(f"benchforge: error: {exc}", file=sys.stderr)
        return 2
    print(format_run_summary(result))
    return 0
