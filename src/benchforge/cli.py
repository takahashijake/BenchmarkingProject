from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from benchforge.analysis.robustness import format_robustness_summary
from benchforge.analysis.summary import (
    format_automl_plan,
    format_automl_summary,
    format_benchmark_summary,
    format_run_summary,
    format_search_summary,
)
from benchforge.automl import build_automl_plan, run_automl
from benchforge.catalog.cli import add_commands, dispatch
from benchforge.core.config import (
    load_automl_config,
    load_benchmark_config,
    load_run_config,
    load_search_config,
)
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.runner import run_benchmark
from benchforge.robustness import load_robustness_config, run_robustness
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
    automl_parser = subparsers.add_parser(
        "automl", help="plan and run compute-aware nested model-family search"
    )
    automl_parser.add_argument("config", help="path to a YAML AutoML configuration")
    automl_parser.add_argument(
        "--plan-only", action="store_true", help="print the deterministic plan without training"
    )
    robustness_parser = subparsers.add_parser(
        "robustness", help="evaluate stability with repeated matched cross-validation"
    )
    robustness_parser.add_argument("config", help="path to a YAML robustness configuration")
    add_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"verify", "catalog"}:
            return dispatch(args)
        if args.command == "run":
            result = run_benchmark(load_run_config(args.config))
            summary = format_run_summary(result)
        elif args.command == "benchmark":
            benchmark_result = run_benchmark_suite(load_benchmark_config(args.config))
            summary = format_benchmark_summary(benchmark_result)
        elif args.command == "search":
            search_result = run_search(load_search_config(args.config))
            summary = format_search_summary(search_result)
        elif args.command == "robustness":
            summary = format_robustness_summary(run_robustness(load_robustness_config(args.config)))
        else:
            automl_config = load_automl_config(args.config)
            if args.plan_only:
                summary = format_automl_plan(build_automl_plan(automl_config))
            else:
                summary = format_automl_summary(run_automl(automl_config))
    except (ValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"benchforge: error: {exc}", file=sys.stderr)
        return 2
    print(summary)
    return 0
