"""Concise terminal interpretation of honest outer decision evaluations."""

from benchforge.decision.results import DecisionResult


def format_decision_summary(result: DecisionResult) -> str:
    lines = [
        f"Decision analysis: {result.fingerprint}",
        f"Status: {result.status}",
        f"Primary metric: {result.primary_metric} ({result.optimization_direction})",
        f"Positive class 1: {result.dataset.target_labels[1]}",
        "Candidate | status | rank | outer mean | per-fold thresholds",
    ]
    for candidate in sorted(
        result.candidates, key=lambda c: (c.rank is None, c.rank or 0, c.identifier)
    ):
        score = next((m.mean for m in candidate.metrics if m.name == result.primary_metric), None)
        thresholds = ", ".join(f"{f.fold_id}:{f.selection.threshold:g}" for f in candidate.folds)
        lines.append(
            f"{candidate.identifier} | {candidate.status} | {candidate.rank} | "
            f"{score} | {thresholds}"
        )
        if candidate.failure is not None:
            lines.append(
                f"  Fold {candidate.failure.fold_id}: {candidate.failure.error_type}: "
                f"{candidate.failure.message}"
            )
        if candidate.folds:
            means = {m.name: m.mean for m in candidate.metrics}
            lines.append(
                "  Selected / reference 0.5 F1: "
                f"{means['selected.f1']:.6f} / {means['reference.f1']:.6f}; "
                f"Brier: {means['probability.brier_score']:.6f}; "
                f"log loss: {means['probability.log_loss']:.6f}"
            )
    lines.extend(
        [
            "Inner scores select thresholds only; leaderboard metrics use held-out outer folds.",
            "Lower thresholds trade precision for recall; costs encode the chosen FP/FN tradeoff.",
            "ECE depends on bins/sample size. "
            "Calibration does not guarantee deployment reliability.",
            "Fold thresholds are separate trained policies, not a universal deployment cutoff.",
            f"Artifacts: {result.artifact_directory}",
        ]
    )
    return "\n".join(lines)
