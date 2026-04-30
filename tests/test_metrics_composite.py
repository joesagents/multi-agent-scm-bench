"""Composite-score math — three normalised components, geometric mean."""

from __future__ import annotations

from scm_bench.metrics.composite import (
    EPSILON,
    ScoreComponents,
    bullwhip_ratio_variance,
    normalize_components,
)


def test_bullwhip_ratio_passthrough_is_one() -> None:
    demand = [4, 8, 4, 8, 4, 8]
    orders = list(demand)
    assert bullwhip_ratio_variance(orders, demand) == 1.0


def test_bullwhip_ratio_amplification_is_greater_than_one() -> None:
    demand = [4, 4, 8, 8, 4, 4]
    orders = [4, 0, 16, 0, 16, 0]
    assert bullwhip_ratio_variance(orders, demand) > 1.0


def test_bullwhip_ratio_smoothing_is_less_than_one() -> None:
    demand = [4, 12, 4, 12, 4, 12]
    orders = [8, 8, 8, 8, 8, 8]
    assert bullwhip_ratio_variance(orders, demand) < 1.0


def test_bullwhip_ratio_flat_demand_returns_one() -> None:
    assert bullwhip_ratio_variance([0, 1, 2, 3], [4, 4, 4, 4]) == 1.0


def test_composite_score_self_normalises_to_one() -> None:
    base = ScoreComponents(bullwhip_ratio=1.0, total_cost=500.0, tokens_used=0.0)
    breakdown = normalize_components(base, base, epsilon=EPSILON)
    assert breakdown.composite_score == 1.0
    assert breakdown.normalized_bullwhip == 1.0
    assert breakdown.normalized_cost == 1.0
    assert breakdown.normalized_tokens == 1.0


def test_composite_score_all_better_drops_below_one() -> None:
    base = ScoreComponents(bullwhip_ratio=2.0, total_cost=1000.0, tokens_used=10.0)
    you = ScoreComponents(bullwhip_ratio=1.0, total_cost=500.0, tokens_used=5.0)
    breakdown = normalize_components(you, base, epsilon=EPSILON)
    assert breakdown.composite_score < 1.0


def test_composite_score_all_worse_climbs_above_one() -> None:
    base = ScoreComponents(bullwhip_ratio=1.0, total_cost=500.0, tokens_used=10.0)
    you = ScoreComponents(bullwhip_ratio=2.0, total_cost=1000.0, tokens_used=20.0)
    breakdown = normalize_components(you, base, epsilon=EPSILON)
    assert breakdown.composite_score > 1.0


def test_composite_score_handles_zero_baseline_tokens() -> None:
    base = ScoreComponents(bullwhip_ratio=1.0, total_cost=500.0, tokens_used=0.0)
    you = ScoreComponents(bullwhip_ratio=1.0, total_cost=500.0, tokens_used=1000.0)
    breakdown = normalize_components(you, base, epsilon=EPSILON)
    # With epsilon=1, normalized_tokens = (1000+1)/(0+1) = 1001 — composite is large
    assert breakdown.normalized_tokens > 100.0
    assert breakdown.composite_score > 1.0
