"""Reusable conservative robustness reporting; no statistical calculations in CLI."""

from benchforge.robustness.results import PairwiseComparison, RobustnessResult


def interpret_pairwise(pair: PairwiseComparison) -> str:
    a, b = pair.candidate_a, pair.candidate_b
    if pair.usable_repetitions < 2:
        return f"Insufficient paired repetitions to assess the ordering of {a} and {b}."
    if pair.ties == pair.usable_repetitions:
        return f"{a} and {b} show no measured separation under this robustness experiment."
    lower, upper = pair.interval.lower_bound, pair.interval.upper_bound
    if lower is not None and lower > 0 and pair.a_win_rate is not None and pair.a_win_rate >= 0.9:
        return f"{a} shows a consistent advantage over {b} across the configured repetitions."
    if upper is not None and upper < 0 and pair.b_win_rate is not None and pair.b_win_rate >= 0.9:
        return f"{b} shows a consistent advantage over {a} across the configured repetitions."
    if pair.mean_delta == 0:
        return f"{a} and {b} have equal paired means, with ordering changes across repetitions."
    winner = a if pair.mean_delta is not None and pair.mean_delta > 0 else b
    return (
        f"{winner} has the better paired mean score, but the ordering is not consistently "
        "supported across the configured repetitions."
    )


def interpret_robustness(result: RobustnessResult) -> str:
    """Keep mean performance, first-place frequency, and rank variability distinct."""
    eligible = [row for row in result.leaderboard if row.status == "success"]
    coverage = "fully completed"
    if not eligible:
        eligible = [row for row in result.leaderboard if row.status == "partial"]
        coverage = "partially completed"
    if not eligible:
        return "No candidate completed a repetition; no performance ordering is available."
    top = eligible[0]
    first_rate = max(row.first_place_rate for row in eligible)
    frequent = ", ".join(
        sorted(row.model_identifier for row in eligible if row.first_place_rate == first_rate)
    )
    rank_std = min(row.rank_std for row in eligible if row.rank_std is not None)
    stable = ", ".join(sorted(row.model_identifier for row in eligible if row.rank_std == rank_std))
    return (
        f"Best mean robustness score among {coverage} candidates: {top.model_identifier}. "
        f"Most frequently first-ranked: {frequent}. "
        f"Lowest rank variability: {stable}. These describe different properties."
    )


def format_pairwise_comparison(pair: PairwiseComparison) -> str:
    if pair.mean_delta is None:
        return f"{pair.candidate_a} vs {pair.candidate_b}: no usable paired repetitions"
    interval = pair.interval
    bounds = "unavailable (fewer than two pairs)"
    if interval.lower_bound is not None and interval.upper_bound is not None:
        bounds = f"[{interval.lower_bound:+.4f}, {interval.upper_bound:+.4f}]"
    return (
        f"{pair.candidate_a} vs {pair.candidate_b} ({pair.usable_repetitions} usable pairs)\n"
        f"  Mean paired delta: {pair.mean_delta:+.4f} (positive means A better)\n"
        f"  {interval.confidence_level:.0%} robustness interval: {bounds}\n"
        f"  A wins / B wins / ties: {pair.a_wins} / {pair.b_wins} / {pair.ties}\n"
        f"  A win rate: {pair.a_win_rate:.0%}\n"
        f"  {interpret_pairwise(pair)}"
    )


def format_robustness_leaderboard(result: RobustnessResult) -> str:
    rows = ["Robustness leaderboard (complete before partial; then best mean score):"]
    for entry in result.leaderboard:
        if entry.primary_metric_mean is None:
            rows.append(
                f"  - {entry.model_identifier}: FAILED; "
                f"{entry.failed_repetitions} failed repetitions"
            )
        else:
            rows.append(
                f"  {entry.rank}. {entry.model_identifier} [{entry.status}] "
                f"mean={entry.primary_metric_mean:.4f} std={entry.primary_metric_std:.4f} "
                f"first={entry.first_place_rate:.0%} mean_rank={entry.mean_rank:.2f}; "
                f"completed={entry.completed_repetitions} failed={entry.failed_repetitions}"
            )
    return "\n".join(rows)


def _rank_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


def format_rank_stability(result: RobustnessResult) -> str:
    return (
        "Rank stability (competition ties; rates over all configured repetitions):\n"
        + "\n".join(
            f"  {item.model_identifier}: "
            f"ranks={{{', '.join(f'{row.rank}: {row.count}' for row in item.rank_counts)}}}; "
            f"mean={_rank_number(item.mean_rank)} median={_rank_number(item.median_rank)} "
            f"std={_rank_number(item.rank_std)}; "
            f"best={item.best_rank} worst={item.worst_rank}; "
            f"ranked={item.ranked_repetitions} missing={item.missing_repetitions}; "
            f"top-2={item.top_2_rate:.0%}"
            for item in result.rank_stability
        )
    )


def format_robustness_failures(result: RobustnessResult) -> str:
    if not result.failures:
        return ""
    return "Candidate failures (incomplete repetitions excluded from comparisons):\n" + "\n".join(
        f"  repetition {item.repetition_id}, {item.model_identifier}, fold {item.fold_id}: "
        f"{item.error_type}: {item.message}; skipped folds {item.skipped_fold_ids}"
        for item in result.failures
    )


def format_robustness_summary(result: RobustnessResult) -> str:
    repetitions = len(result.plan.repetitions)
    folds = len(result.plan.repetitions[0].folds)
    lines = [
        "BenchForge robustness",
        f"Task: {result.dataset.task.value}",
        f"Dataset: {result.dataset.identity}",
        f"Candidates: {len(result.plan.candidate_identifiers)}",
        f"Repetitions: {repetitions}",
        f"Folds/repetition: {folds}",
        f"Primary metric: {result.primary_metric} ({result.optimization_direction})",
        f"Approximate maximum estimator fits: {result.plan.approximate_maximum_fits} "
        f"({repetitions} × {len(result.plan.candidate_identifiers)} × {folds})",
        format_robustness_leaderboard(result),
        format_rank_stability(result),
        interpret_robustness(result),
        "Pairwise evidence:",
        *(format_pairwise_comparison(pair) for pair in result.pairwise_comparisons),
    ]
    if result.failures:
        lines.extend(
            [
                format_robustness_failures(result),
                "Ranks compare available successful candidates; failures may bias summaries.",
            ]
        )
    lines.extend(
        [
            "Intervals describe fold/estimator resampling sensitivity on this dataset; "
            "they do not establish population uncertainty or statistical significance.",
            f"Fingerprint: {result.fingerprint}",
            f"Artifacts: {result.artifact_directory or 'not persisted'}",
        ]
    )
    return "\n".join(lines)
