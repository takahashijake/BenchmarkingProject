from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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
from benchforge.execution.supervisor import ExecutionOptions
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
    benchmark_parser.add_argument(
        "--workers", type=int, default=1, help="parallel candidate processes (default: 1)"
    )
    benchmark_parser.add_argument("--candidate-timeout", type=float, help="candidate wall seconds")
    benchmark_parser.add_argument("--run-timeout", type=float, help="run execution wall seconds")
    workspace_args = benchmark_parser.add_mutually_exclusive_group()
    workspace_args.add_argument(
        "--workspace", type=Path, help="create a new candidate checkpoint workspace"
    )
    workspace_args.add_argument(
        "--resume", type=Path, help="resume a previously interrupted workspace"
    )
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
    for recovery_parser in (search_parser, automl_parser):
        recovery_parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="maximum family/outer-fold processes (default: 1)",
        )
        recovery_args = recovery_parser.add_mutually_exclusive_group()
        recovery_args.add_argument(
            "--workspace", type=Path, help="create a complete outer-fold checkpoint workspace"
        )
        recovery_args.add_argument(
            "--resume", type=Path, help="reuse verified completed outer-fold tasks"
        )
    robustness_parser = subparsers.add_parser(
        "robustness", help="evaluate stability with repeated matched cross-validation"
    )
    robustness_parser.add_argument("config", help="path to a YAML robustness configuration")
    report_parser = subparsers.add_parser(
        "report", help="render verified artifacts as offline HTML"
    )
    report_parser.add_argument("directory", type=Path)
    report_parser.add_argument("--output", required=True, type=Path)
    report_parser.add_argument("--compare", type=Path, help="second artifact directory")
    report_parser.add_argument(
        "--deep", action="store_true", help="recompute stored scores offline"
    )
    report_parser.add_argument(
        "--allow-legacy", action="store_true", help="permit verified historical unsealed evidence"
    )
    report_parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing report"
    )
    from benchforge.providers.cli import add_commands as add_plugin_commands

    add_plugin_commands(subparsers)
    add_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            from benchforge.reporting import generate_report

            output = generate_report(
                args.directory,
                args.output,
                compare=args.compare,
                deep=args.deep,
                allow_legacy=args.allow_legacy,
                overwrite=args.overwrite,
            )
            print(f"Report: {output}")
            return 0
        if args.command == "plugins":
            from benchforge.providers.cli import dispatch as dispatch_plugins

            return dispatch_plugins(args)
        if args.command in {"verify", "catalog"}:
            return dispatch(args)
        if args.command == "run":
            result = run_benchmark(load_run_config(args.config))
            summary = format_run_summary(result)
        elif args.command == "benchmark":
            benchmark_result = run_benchmark_suite(
                load_benchmark_config(args.config),
                workers=args.workers,
                execution_options=ExecutionOptions(args.candidate_timeout, args.run_timeout),
                checkpoint_dir=args.workspace or args.resume,
                resume=args.resume is not None,
            )
            summary = format_benchmark_summary(benchmark_result)
        elif args.command == "search":
            search_result = run_search(
                load_search_config(args.config),
                workers=args.workers,
                checkpoint_dir=args.workspace or args.resume,
                resume=args.resume is not None,
            )
            summary = format_search_summary(search_result)
        elif args.command == "robustness":
            summary = format_robustness_summary(run_robustness(load_robustness_config(args.config)))
        else:
            automl_config = load_automl_config(args.config)
            if args.plan_only:
                from benchforge.search.parallel import validate_workers

                validate_workers(args.workers)
                if args.workspace or args.resume:
                    raise ValueError("--plan-only cannot be combined with workspace recovery")
                summary = format_automl_plan(build_automl_plan(automl_config))
            else:
                automl_result = run_automl(
                    automl_config,
                    workers=args.workers,
                    checkpoint_dir=args.workspace or args.resume,
                    resume=args.resume is not None,
                )
                summary = format_automl_summary(automl_result)
        if args.command in {"search", "automl"} and not (
            args.command == "automl" and args.plan_only
        ):
            recovered_search = (
                search_result if args.command == "search" else automl_result.search_result
            )
            outcomes = recovered_search.execution_diagnostics
            completed = sum(d.outcome == "success" for d in outcomes)
            reused = sum(d.outcome == "reused" for d in outcomes)
            failed = sum(d.outcome == "failed" for d in outcomes)
            summary += (
                f"\nExecution mode: {recovered_search.execution_mode} "
                f"({recovered_search.execution_workers} workers)"
                f"\nPlanned tasks: {recovered_search.execution_planned_tasks}"
                f"\nReused tasks: {reused}"
                f"\nExecuted tasks: {completed}"
                f"\nFailed tasks: {failed}"
                f"\nInvalid checkpoints: {len(recovered_search.checkpoint_invalid)}"
                f"\nRemaining tasks: "
                f"{recovered_search.execution_planned_tasks - completed - reused - failed}"
            )
    except KeyboardInterrupt:
        print("benchforge: interrupted; committed checkpoints can be resumed", file=sys.stderr)
        return 130
    except (ValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"benchforge: error: {exc}", file=sys.stderr)
        return 2
    print(summary)
    return 0
