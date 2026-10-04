from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from benchforge.analysis.robustness import interpret_pairwise
from benchforge.robustness import RobustnessSettings
from benchforge.robustness.results import CandidateScore, RepetitionScores
from benchforge.robustness.statistics import (
    bootstrap_interval,
    compare_all_pairs,
    compare_pair,
    rank_repetition,
    summarize_ranks,
)


def observations(values: list[tuple[float, float]]) -> tuple[RepetitionScores, ...]:
    return tuple(
        RepetitionScores(index, (CandidateScore("a", a), CandidateScore("b", b)))
        for index, (a, b) in enumerate(values)
    )


@pytest.mark.parametrize("direction, sign", [("maximize", 1), ("minimize", -1)])
def test_paired_math_and_direction(direction: str, sign: int) -> None:
    evidence = observations([(4, 1), (1, 2), (2, 2), (5, 3)])
    pair = compare_pair("a", "b", evidence, "metric", direction, RobustnessSettings(), 42)
    expected = np.array([3, -1, 0, 2]) * sign
    assert pair.raw_paired_deltas == (3, -1, 0, 2)
    assert pair.paired_deltas == tuple(expected)
    assert pair.mean_delta == pytest.approx(expected.mean())
    assert pair.median_delta == pytest.approx(np.median(expected))
    assert pair.std_delta == pytest.approx(expected.std(ddof=1))
    assert pair.minimum_delta == min(expected)
    assert pair.maximum_delta == max(expected)
    assert pair.ties == 1
    assert pair.a_wins == (2 if sign == 1 else 1)
    assert pair.b_wins == (1 if sign == 1 else 2)
    assert pair.a_win_rate == pair.a_wins / 4
    assert pair.b_win_rate == pair.b_wins / 4
    assert pair.repetition_ids == (0, 1, 2, 3)
    assert pair == compare_pair("a", "b", evidence, "metric", direction, RobustnessSettings(), 42)
    with pytest.raises(FrozenInstanceError):
        pair.mean_delta = 0


def test_bootstrap_matches_numpy_reference_and_constant_effect() -> None:
    settings = RobustnessSettings(bootstrap_samples=2000)
    deltas = (1.0, 2.0, -1.0, 0.0)
    interval = bootstrap_interval(deltas, settings, 15)
    rng = np.random.Generator(np.random.PCG64(15))
    values = np.array(deltas)[rng.integers(0, 4, size=(2000, 4))].mean(axis=1)
    assert interval.lower_bound == pytest.approx(np.quantile(values, 0.025))
    assert interval.upper_bound == pytest.approx(np.quantile(values, 0.975))
    constant = bootstrap_interval((0.1, 0.1, 0.1), settings, 42)
    assert constant.lower_bound == pytest.approx(0.1)
    assert constant.upper_bound == pytest.approx(0.1)
    assert bootstrap_interval(deltas, settings, 15) == interval


def test_missing_pairs_remain_visible_and_no_singleton_interval() -> None:
    evidence = (
        RepetitionScores(0, (CandidateScore("a", 1),)),
        RepetitionScores(1, (CandidateScore("a", 2), CandidateScore("b", 1))),
        RepetitionScores(2, ()),
    )
    pairs = compare_all_pairs(
        ("c", "b", "a"), evidence, "accuracy", "maximize", RobustnessSettings(), 1
    )
    assert [(pair.candidate_a, pair.candidate_b) for pair in pairs] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert pairs[0].usable_repetitions == 1
    assert pairs[0].repetition_ids == (1,)
    assert pairs[0].interval.lower_bound is None
    assert pairs[1].usable_repetitions == 0
    assert pairs[1].mean_delta is None
    assert pairs[1].a_win_rate is None
    assert "Insufficient" in interpret_pairwise(pairs[0])


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_competition_ranks_do_not_depend_on_insertion_order(direction: str) -> None:
    scores = RepetitionScores(
        0, (CandidateScore("c", 1), CandidateScore("a", 2), CandidateScore("b", 2))
    )
    ranked = rank_repetition(scores, direction)
    assert ranked == rank_repetition(RepetitionScores(0, tuple(reversed(scores.scores))), direction)
    expected = {"a": 1, "b": 1, "c": 3} if direction == "maximize" else {"a": 2, "b": 2, "c": 1}
    assert {row.model_identifier: row.rank for row in ranked} == expected


def test_rank_distribution_rates_and_missing_denominators() -> None:
    evidence = (*observations([(2, 1), (1, 2), (1, 1)]), RepetitionScores(3, ()))
    a, b, c = summarize_ranks(("c", "b", "a"), evidence, "maximize")
    assert a.model_identifier == "a"
    assert a.mean_rank == pytest.approx(4 / 3)
    assert a.median_rank == 1
    assert a.rank_std == pytest.approx(np.std([1, 2, 1], ddof=1))
    assert a.best_rank == 1 and a.worst_rank == 2
    assert a.first_place_count == 2 and a.first_place_rate == 0.5
    assert a.top_2_count == 3 and a.top_2_rate == 0.75
    assert a.ranked_repetitions == 3 and a.missing_repetitions == 1
    assert {row.rank: row.count for row in a.rank_counts} == {1: 2, 2: 1, 3: 0}
    assert sum(row.count for row in b.rank_counts) == 3
    assert c.ranked_repetitions == 0 and c.mean_rank is None and c.first_place_rate == 0


def test_conservative_interpretations() -> None:
    settings = RobustnessSettings()

    def pair(values: list[tuple[float, float]]):
        return compare_pair("a", "b", observations(values), "accuracy", "maximize", settings, 42)

    assert "consistent advantage" in interpret_pairwise(pair([(2, 1)] * 5))
    assert "b shows a consistent advantage" in interpret_pairwise(pair([(1, 2)] * 5))
    assert "no measured separation" in interpret_pairwise(pair([(1, 1)] * 5))
    assert "not consistently" in interpret_pairwise(pair([(3, 1), (1, 2)] * 5))


def test_reversing_pair_preserves_bootstrap_draws_and_reverses_effect() -> None:
    evidence = observations([(4, 1), (1, 2), (2, 2), (5, 3)])
    a = compare_pair("a", "b", evidence, "accuracy", "maximize", RobustnessSettings(), 42)
    b = compare_pair("b", "a", evidence, "accuracy", "maximize", RobustnessSettings(), 42)
    assert a.interval.seed == b.interval.seed
    assert a.mean_delta == -b.mean_delta
    assert a.interval.lower_bound == pytest.approx(-b.interval.upper_bound)
    assert a.interval.upper_bound == pytest.approx(-b.interval.lower_bound)
    assert a.a_wins == b.b_wins
    equal_mean = compare_pair(
        "a", "b", observations([(2, 1), (1, 2)]), "accuracy", "maximize", RobustnessSettings(), 42
    )
    assert "equal paired means" in interpret_pairwise(equal_mean)
