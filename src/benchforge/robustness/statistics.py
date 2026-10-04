"""Descriptive paired repetition analysis, with no fold-level hypothesis tests."""

from itertools import combinations

import numpy as np

from benchforge.core.seeds import derive_seed
from benchforge.robustness.config import RobustnessSettings
from benchforge.robustness.results import (
    CandidateRank,
    ConfidenceInterval,
    Direction,
    PairwiseComparison,
    RankFrequency,
    RankStability,
    RepetitionScores,
)


def sample_std(values: tuple[float, ...]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bootstrap_interval(
    deltas: tuple[float, ...], settings: RobustnessSettings, seed: int
) -> ConfidenceInterval:
    """Percentile bootstrap of paired repetition means, conditional on this dataset.

    Repetitions reuse rows and are not independent datasets. This interval describes
    resampling sensitivity; it is not a population confidence or significance claim.
    Fewer than two paired repetitions cannot support an interval.
    """
    if len(deltas) < 2:
        return ConfidenceInterval(
            settings.confidence_level, None, None, settings.bootstrap_samples, seed
        )
    values = np.asarray(deltas, dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(settings.bootstrap_samples, dtype=float)
    # Bound temporary index arrays to about one million elements, even for many repetitions.
    batch_size = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, settings.bootstrap_samples, batch_size):
        stop = min(settings.bootstrap_samples, start + batch_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1 - settings.confidence_level) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail], method="linear")
    return ConfidenceInterval(
        settings.confidence_level, float(lower), float(upper), settings.bootstrap_samples, seed
    )


def compare_pair(
    candidate_a: str,
    candidate_b: str,
    observations: tuple[RepetitionScores, ...],
    primary_metric: str,
    direction: Direction,
    settings: RobustnessSettings,
    seed: int,
) -> PairwiseComparison:
    ids: list[int] = []
    raw: list[float] = []
    for repetition in sorted(observations, key=lambda item: item.repetition_id):
        scores = {item.model_identifier: item.value for item in repetition.scores}
        if candidate_a in scores and candidate_b in scores:
            ids.append(repetition.repetition_id)
            raw.append(scores[candidate_a] - scores[candidate_b])
    deltas = tuple(value if direction == "maximize" else -value for value in raw)
    a_wins = sum(value > 0 for value in deltas)
    b_wins = sum(value < 0 for value in deltas)
    count = len(deltas)
    interval_seed = derive_seed(
        seed, "robustness", "bootstrap", *sorted((candidate_a, candidate_b))
    )
    return PairwiseComparison(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        primary_metric=primary_metric,
        optimization_direction=direction,
        usable_repetitions=count,
        repetition_ids=tuple(ids),
        raw_paired_deltas=tuple(raw),
        paired_deltas=deltas,
        mean_delta=float(np.mean(deltas)) if count else None,
        median_delta=float(np.median(deltas)) if count else None,
        std_delta=sample_std(deltas) if count else None,
        minimum_delta=min(deltas) if count else None,
        maximum_delta=max(deltas) if count else None,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=count - a_wins - b_wins,
        a_win_rate=a_wins / count if count else None,
        b_win_rate=b_wins / count if count else None,
        interval=bootstrap_interval(deltas, settings, interval_seed),
    )


def compare_all_pairs(
    identifiers: tuple[str, ...],
    observations: tuple[RepetitionScores, ...],
    primary_metric: str,
    direction: Direction,
    settings: RobustnessSettings,
    seed: int,
) -> tuple[PairwiseComparison, ...]:
    return tuple(
        compare_pair(a, b, observations, primary_metric, direction, settings, seed)
        for a, b in combinations(sorted(identifiers), 2)
    )


def rank_repetition(scores: RepetitionScores, direction: Direction) -> tuple[CandidateRank, ...]:
    """Exact ties share competition rank: 1, 1, 3; identifiers only order the output."""
    return tuple(
        CandidateRank(
            candidate.model_identifier,
            1
            + sum(
                other.value > candidate.value
                if direction == "maximize"
                else other.value < candidate.value
                for other in scores.scores
            ),
        )
        for candidate in sorted(scores.scores, key=lambda item: item.model_identifier)
    )


def summarize_ranks(
    identifiers: tuple[str, ...],
    observations: tuple[RepetitionScores, ...],
    direction: Direction,
) -> tuple[RankStability, ...]:
    ranked = tuple(rank_repetition(repetition, direction) for repetition in observations)
    total = len(observations)
    results = []
    for identifier in sorted(identifiers):
        ranks = tuple(
            float(row.rank)
            for repetition in ranked
            for row in repetition
            if row.model_identifier == identifier
        )
        first = sum(rank == 1 for rank in ranks)
        top2 = sum(rank <= 2 for rank in ranks)
        results.append(
            RankStability(
                model_identifier=identifier,
                ranked_repetitions=len(ranks),
                missing_repetitions=total - len(ranks),
                mean_rank=float(np.mean(ranks)) if ranks else None,
                median_rank=float(np.median(ranks)) if ranks else None,
                rank_std=sample_std(ranks) if ranks else None,
                best_rank=int(min(ranks)) if ranks else None,
                worst_rank=int(max(ranks)) if ranks else None,
                first_place_count=first,
                first_place_rate=first / total,
                top_2_count=top2,
                top_2_rate=top2 / total,
                rank_counts=tuple(
                    RankFrequency(rank, ranks.count(rank))
                    for rank in range(1, len(identifiers) + 1)
                ),
            )
        )
    return tuple(results)
