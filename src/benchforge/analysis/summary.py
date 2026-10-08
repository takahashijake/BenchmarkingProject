from benchforge.automl.planner import AutoMLPlan
from benchforge.automl.results import AutoMLResult
from benchforge.core.config import MetricName
from benchforge.evaluation.metrics import metric_spec
from benchforge.execution.benchmark import BenchmarkResult
from benchforge.execution.runner import RunResult
from benchforge.search.results import SearchResult


def format_run_summary(result: RunResult) -> str:
    metric_lines = "\n".join(
        f"  {metric_spec(MetricName(name)).display_name} "
        f"{'↑' if metric_spec(MetricName(name)).direction == 'maximize' else '↓'}: "
        f"{value.mean:.4f} ± {value.std:.4f}"
        for name, value in result.aggregate_metrics.items()
    )
    artifact = str(result.artifact_directory) if result.artifact_directory else "not persisted"
    return (
        "BenchForge run complete\n"
        f"Dataset: {result.dataset_identity}\n"
        f"Model: {result.model_name}\n"
        f"Folds: {len(result.fold_results)}\n"
        f"Metrics:\n{metric_lines}\n"
        f"Fingerprint: {result.fingerprint}\n"
        f"Artifacts: {artifact}"
    )


def format_benchmark_summary(result: BenchmarkResult) -> str:
    spec = metric_spec(MetricName(result.primary_metric))
    rows = []
    for entry in result.leaderboard:
        if entry.status == "success":
            if entry.primary_metric_mean is None or entry.primary_metric_std is None:
                raise ValueError("successful leaderboard entry is missing primary metric values")
            rows.append(
                f"  {entry.rank}. {entry.model_identifier}: "
                f"{entry.primary_metric_mean:.4f} ± {entry.primary_metric_std:.4f} "
                f"({entry.duration_seconds:.3f}s)"
            )
        else:
            rows.append(f"  - {entry.model_identifier}: FAILED ({entry.duration_seconds:.3f}s)")
    failures = ""
    if result.failures:
        failures = "\nCandidate failures:\n" + "\n".join(
            f"  {failure.model_identifier}: {failure.error_type}: {failure.message}"
            for failure in result.failures
        )
    artifact = str(result.artifact_directory) if result.artifact_directory else "not persisted"
    recovery = ""
    if result.checkpoint_workspace is not None:
        recovery = (
            f"\nCheckpoint workspace: {result.checkpoint_workspace}"
            f"\nReused candidates: {len(result.checkpoint_reused)} "
            f"({', '.join(result.checkpoint_reused) or 'none'})"
            f"\nNewly evaluated candidates: {len(result.checkpoint_executed)}"
        )
        if result.checkpoint_invalid:
            recovery += "\nRejected checkpoints: " + "; ".join(result.checkpoint_invalid)
    return (
        "BenchForge benchmark complete\n"
        f"Dataset: {result.dataset.identity}\n"
        f"Shape: {result.dataset.row_count} rows × {result.dataset.feature_count} features\n"
        f"Folds: {result.fold_count}\n"
        f"Primary metric: {spec.display_name} "
        f"({'higher' if spec.direction == 'maximize' else 'lower'} is better)\n"
        f"Leaderboard ({'highest' if spec.direction == 'maximize' else 'lowest'} "
        "measured mean first):\n"
        + "\n".join(rows)
        + failures
        + f"\nFingerprint: {result.fingerprint}\n"
        f"Artifacts: {artifact}" + recovery
    )


def format_search_summary(result: SearchResult) -> str:
    spec = metric_spec(MetricName(result.primary_metric))
    rows = []
    for entry in result.leaderboard:
        if entry.status == "success":
            if entry.primary_metric_mean is None or entry.primary_metric_std is None:
                raise ValueError("successful leaderboard entry is missing primary metric values")
            rows.append(
                f"  {entry.rank}. {entry.model_identifier} ({entry.mode}): "
                f"{entry.primary_metric_mean:.4f} ± {entry.primary_metric_std:.4f}; "
                f"trials {entry.completed_trials}/{entry.configured_trials} completed, "
                f"{entry.failed_trials} failed ({entry.duration_seconds:.3f}s)"
            )
        else:
            rows.append(f"  - {entry.model_identifier}: FAILED ({entry.duration_seconds:.3f}s)")
    successful = [entry for entry in result.leaderboard if entry.status == "success"]
    top = successful[0]
    final = "\nFinal full-data search: disabled"
    if result.final_candidate is not None:
        candidate = result.final_candidate
        final = (
            "\nFinal full-data search (selection evidence, not held-out performance):\n"
            f"  family: {candidate.model_identifier}\n"
            f"  best params: {candidate.parameters}\n"
            f"  CV selection score: {candidate.cv_selection_score:.4f}\n"
            f"  trials: {candidate.completed_trials}/{candidate.configured_trials} completed"
        )
    failures = ""
    if result.failures:
        failures = "\nFamily failures:\n" + "\n".join(
            f"  {failure.model_identifier}: {failure.error_type}: {failure.message}"
            for failure in result.failures
        )
    artifact = str(result.artifact_directory) if result.artifact_directory else "not persisted"
    return (
        "BenchForge nested search complete\n"
        f"Dataset: {result.dataset.identity}\n"
        f"Outer folds: {result.outer_fold_count}\n"
        f"Inner folds: {result.inner_fold_count}\n"
        f"Optimization metric: {spec.display_name} "
        f"({'higher' if spec.direction == 'maximize' else 'lower'} is better; "
        f"{result.optimization_direction})\n"
        "Tuned-family leaderboard (outer-fold held-out means only):\n"
        + "\n".join(rows)
        + failures
        + (
            f"\nTop measured tuned family: {top.model_identifier}"
            if spec.direction == "maximize"
            else f"\nTop measured result under {spec.display_name}: {top.model_identifier}"
        )
        + final
        + f"\nNested evaluation estimate: {result.primary_metric} "
        f"{top.primary_metric_mean:.4f} ± {top.primary_metric_std:.4f}\n"
        f"Total duration: {result.total_duration_seconds:.3f}s\n"
        f"Search fingerprint: {result.fingerprint}\n"
        f"Artifacts: {artifact}"
    )


def format_automl_plan(plan: AutoMLPlan) -> str:
    searchable = ", ".join(plan.searchable_families)
    fixed = ", ".join(plan.fixed_families) if plan.fixed_families else "none"
    timeout_note = ""
    if plan.search_config.budget.timeout_seconds_per_outer_fold is not None:
        timeout_note = (
            "\nTrial counts are upper bounds because an outer-study timeout is configured."
        )
    return (
        "BenchForge AutoML plan\n"
        f"Dataset: {plan.dataset_identity}\n"
        f"Task: {plan.search_config.task.value}\n"
        f"Searchable families ({len(plan.searchable_families)}): {searchable}\n"
        f"Fixed candidates ({len(plan.fixed_families)}): {fixed}\n"
        f"Outer folds: {plan.outer_fold_count}\n"
        f"Inner folds: {plan.inner_fold_count}\n"
        f"Total trial budget: {plan.total_trial_budget}\n"
        f"Final-search reserve: {plan.final_search_trials}\n"
        f"Outer-search budget: {plan.outer_search_budget}\n"
        f"Family/fold studies: {plan.study_count}\n"
        f"Trials per family / outer fold: {plan.trials_per_outer_fold}\n"
        f"Allocated outer trials: {plan.allocated_outer_trials}\n"
        f"Allocated trials including final reserve: {plan.allocated_trials}\n"
        f"Unused trials: {plan.unallocated_trials}\n"
        f"Approximate maximum estimator fits: {plan.approximate_maximum_fits}\n"
        f"AutoML fingerprint: {plan.fingerprint}\n"
        f"Generated search fingerprint: {plan.search_fingerprint}"
        + timeout_note
    )


def format_automl_summary(result: AutoMLResult) -> str:
    search = result.search_result
    spec = metric_spec(MetricName(search.primary_metric))
    top = next(entry for entry in search.leaderboard if entry.status == "success")
    final = "disabled"
    if search.final_candidate is not None:
        final = (
            f"{search.final_candidate.model_identifier}; parameters "
            f"{search.final_candidate.parameters}"
        )
    artifact = str(result.artifact_directory) if result.artifact_directory else "not persisted"
    return (
        "BenchForge AutoML complete\n"
        f"Dataset: {search.dataset.identity}\n"
        f"Task: {search.dataset.task.value}\n"
        f"Searchable families: {', '.join(result.plan.searchable_families)}\n"
        f"Fixed candidates: {', '.join(result.plan.fixed_families) or 'none'}\n"
        f"Budget: {result.plan.allocated_outer_trials} outer trials allocated, "
        f"{result.plan.unallocated_trials} unused, "
        f"{result.plan.final_search_trials} reserved for final search\n"
        f"Primary metric: {spec.display_name} ({search.optimization_direction})\n"
        f"Top measured family: {top.model_identifier}\n"
        f"Nested-CV result: {top.primary_metric_mean:.4f} ± {top.primary_metric_std:.4f}\n"
        f"Final full-data candidate: {final}\n"
        f"Total duration: {result.total_duration_seconds:.3f}s\n"
        f"AutoML fingerprint: {result.fingerprint}\n"
        f"Search fingerprint: {search.fingerprint}\n"
        f"Artifacts: {artifact}"
    )
