"""Scenarios — demand-fn determinism, named scenario contract, registry."""

from __future__ import annotations

import pytest

from scm_bench.scenarios.builtin import (
    PUBLIC_LEVELS,
    INTRO_STEP_DEMAND,
    S1_1_STABLE,
    S2_3_DEMAND_SHOCK,
    SCENARIOS,
    get,
    list_ids,
)
from scm_bench.scenarios.library import (
    constant,
    iid_uniform,
    scripted,
    seasonal,
    step,
)


def test_constant_demand_is_period_independent() -> None:
    fn = constant(7)
    for p in range(20):
        assert fn(p, seed=42) == 7
    # seed must not change a constant
    assert fn(0, seed=0) == fn(0, seed=999)


def test_step_demand_switches_at_threshold() -> None:
    fn = step(low=4, high=8, switch_at=5)
    assert [fn(p, seed=0) for p in range(8)] == [4, 4, 4, 4, 4, 8, 8, 8]


def test_iid_uniform_deterministic_per_seed() -> None:
    fn = iid_uniform(low=1, high=10)
    seq_a = [fn(p, seed=123) for p in range(50)]
    seq_b = [fn(p, seed=123) for p in range(50)]
    assert seq_a == seq_b
    seq_c = [fn(p, seed=124) for p in range(50)]
    # different seeds should diverge somewhere in 50 draws
    assert seq_a != seq_c


def test_iid_uniform_respects_bounds() -> None:
    fn = iid_uniform(low=2, high=5)
    for p in range(200):
        v = fn(p, seed=7)
        assert 2 <= v <= 5


def test_seasonal_demand_is_non_negative_and_periodic() -> None:
    fn = seasonal(base=10, amplitude=8, cycle=12)
    one_cycle = [fn(p, seed=0) for p in range(12)]
    next_cycle = [fn(p, seed=0) for p in range(12, 24)]
    assert one_cycle == next_cycle
    assert all(v >= 0 for v in one_cycle)


def test_scripted_demand_clamps_past_end() -> None:
    fn = scripted([2, 4, 6, 8])
    assert [fn(p, seed=0) for p in range(6)] == [2, 4, 6, 8, 8, 8]


def test_get_returns_named_scenarios() -> None:
    assert get("intro_step_demand") is INTRO_STEP_DEMAND
    assert get("s1.1") is S1_1_STABLE
    assert get("s2.3") is S2_3_DEMAND_SHOCK


def test_get_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        get("s99.9")


def test_list_ids_includes_all_public_levels_and_intro() -> None:
    ids = list_ids()
    assert "intro_step_demand" in ids
    for level in PUBLIC_LEVELS:
        assert level in ids
    assert ids == sorted(SCENARIOS)


def test_public_levels_have_full_year_horizon() -> None:
    for level in PUBLIC_LEVELS:
        assert get(level).env_config.horizon == 365


def test_intro_step_demand_short_horizon_for_smoke() -> None:
    assert INTRO_STEP_DEMAND.env_config.horizon == 30


def test_s2_3_demand_jumps_at_period_180() -> None:
    fn = S2_3_DEMAND_SHOCK.demand_fn
    assert fn(179, seed=0) == 5
    assert fn(180, seed=0) == 12
    assert fn(364, seed=0) == 12


def test_named_scenarios_share_baseline_costs() -> None:
    """All three benchmark levels use the canonical course cost structure."""
    for level in PUBLIC_LEVELS:
        env = get(level).env_config
        assert env.holding_cost == 0.50
        assert env.backlog_cost == 1.00
