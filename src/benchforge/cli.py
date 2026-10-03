from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from benchforge.analysis.summary import (
    format_benchmark_summary,
    format_run_summary,
    format_search_summary,
)
from benchforge.core.config import load_benchmark_config, load_run_config, load_search_config
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.runner import run_benchmark
from benchforge.search.runner import run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchforge", description="Run reproducible classical-ML benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute a benchmark configuration")
    run_parser.add_argument("config", help="path to a YAML run configuration")
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="compare multiple models on shared cross-validation folds"
    )
    benchmark_parser.add_argument("config", help="path to a YAML benchmark configuration")
    search_parser = subparsers.add_parser(
        "search", help="tune model families with nested cross-validation"
    )
    search_parser.add_argument("config", help="path to a YAML search configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_benchmark(load_run_config(args.config))
            summary = format_run_summary(result)
        elif args.command == "benchmark":
            benchmark_result = run_benchmark_suite(load_benchmark_config(args.config))
            summary = format_benchmark_summary(benchmark_result)
        else:
            search_result = run_search(load_search_config(args.config))
            summary = format_search_summary(search_result)
    except (ValidationError, ValueError, RuntimeError) as exc:
        print(f"benchforge: error: {exc}", file=sys.stderr)
        return 2
    print(summary)
    return 0
