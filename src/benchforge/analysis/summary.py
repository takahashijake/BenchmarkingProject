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
