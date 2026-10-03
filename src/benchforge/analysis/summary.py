from benchforge.execution.benchmark import BenchmarkResult
from benchforge.execution.runner import RunResult


def format_run_summary(result: RunResult) -> str:
    metric_lines = "\n".join(
        f"  {name}: {value.mean:.4f} ± {value.std:.4f}"
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
    return (
        "BenchForge benchmark complete\n"
        f"Dataset: {result.dataset.identity}\n"
        f"Shape: {result.dataset.row_count} rows × {result.dataset.feature_count} features\n"
        f"Folds: {result.fold_count}\n"
        f"Primary metric: {result.primary_metric}\n"
        "Leaderboard (highest measured mean first):\n"
        + "\n".join(rows)
        + failures
        + f"\nFingerprint: {result.fingerprint}\n"
        f"Artifacts: {artifact}"
    )
