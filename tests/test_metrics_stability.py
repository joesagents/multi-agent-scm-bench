"""Stability check — quartile match across S1.1 and S2.3."""

from __future__ import annotations

from scm_bench.metrics.stability import direction_held, quartile


def test_quartile_lowest_value_is_q1() -> None:
    distribution = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8]
    assert quartile(0.5, distribution) == 1


def test_quartile_highest_value_is_q4() -> None:
    distribution = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8]
    assert quartile(1.8, distribution) == 4


def test_quartile_singleton_is_q1() -> None:
    assert quartile(1.0, [1.0]) == 1


def test_quartile_empty_distribution_is_zero() -> None:
    assert quartile(1.0, []) == 0


def test_direction_held_when_team_stays_in_top_quartile() -> None:
    population_stable = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    population_disrupted = [0.6, 0.7, 0.9, 1.1, 1.3, 1.6]
    assert direction_held(
        team_score_stable=0.5,
        team_score_disrupted=0.6,
        population_scores_stable=population_stable,
        population_scores_disrupted=population_disrupted,
    )


def test_direction_held_false_when_team_drops_quartile() -> None:
    population_stable = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    population_disrupted = [0.6, 0.7, 0.9, 1.1, 1.3, 1.6]
    assert not direction_held(
        team_score_stable=0.5,    # quartile 1
        team_score_disrupted=1.5, # quartile 4
        population_scores_stable=population_stable,
        population_scores_disrupted=population_disrupted,
    )
