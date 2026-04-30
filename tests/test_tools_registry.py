"""Tool registry + builtin tools — registration, lookup, math."""

from __future__ import annotations

import pytest

from scm_bench.tools.builtin import (
    BUILTIN_TOOLS,
    default_registry,
    forecast_exponential_smoothing,
    forecast_moving_average,
    local_cost_estimator,
)
from scm_bench.tools.registry import ToolRegistry


def test_register_and_get_round_trip() -> None:
    reg = ToolRegistry()
    reg.register("noop", lambda: None)
    assert "noop" in reg
    assert callable(reg.get("noop"))


def test_register_duplicate_name_raises() -> None:
    reg = ToolRegistry()
    reg.register("ma", forecast_moving_average)
    with pytest.raises(ValueError):
        reg.register("ma", forecast_moving_average)


def test_get_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_names_returns_sorted_list() -> None:
    reg = ToolRegistry()
    reg.register("zeta", lambda: None)
    reg.register("alpha", lambda: None)
    reg.register("mu", lambda: None)
    assert reg.names() == ["alpha", "mu", "zeta"]


def test_default_registry_contains_three_builtins() -> None:
    reg = default_registry()
    assert set(reg.names()) == {
        "forecast_moving_average",
        "forecast_exponential_smoothing",
        "local_cost_estimator",
    }
    for name in BUILTIN_TOOLS:
        assert name in reg


def test_default_registry_returns_fresh_instance() -> None:
    """Two calls return independent registries (mutating one does not affect the other)."""
    a = default_registry()
    b = default_registry()
    a.register("custom", lambda: None)
    assert "custom" not in b


def test_moving_average_empty_returns_zero() -> None:
    assert forecast_moving_average([]) == 0.0


def test_moving_average_window_clamps_to_length() -> None:
    assert forecast_moving_average([4, 6], window=10) == 5.0


def test_moving_average_uses_tail_window() -> None:
    assert forecast_moving_average([1, 1, 1, 100, 100, 100], window=3) == 100.0


def test_moving_average_window_at_least_one() -> None:
    assert forecast_moving_average([10, 20, 30], window=0) == 30.0


def test_exponential_smoothing_empty_returns_zero() -> None:
    assert forecast_exponential_smoothing([]) == 0.0


def test_exponential_smoothing_alpha_one_returns_last() -> None:
    assert forecast_exponential_smoothing([1, 2, 3, 99], alpha=1.0) == 99.0


def test_exponential_smoothing_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        forecast_exponential_smoothing([1, 2, 3], alpha=0.0)
    with pytest.raises(ValueError):
        forecast_exponential_smoothing([1, 2, 3], alpha=1.5)


def test_exponential_smoothing_seeded_value_for_known_input() -> None:
    # alpha=0.5 gives easy hand-checkable arithmetic
    # s0=2, s1=0.5*4 + 0.5*2 = 3, s2=0.5*6 + 0.5*3 = 4.5
    assert forecast_exponential_smoothing([2, 4, 6], alpha=0.5) == 4.5


def test_local_cost_estimator_default_weights() -> None:
    assert local_cost_estimator(inventory=10, backlog=2) == 10 * 0.50 + 2 * 1.00


def test_local_cost_estimator_custom_weights() -> None:
    assert (
        local_cost_estimator(
            inventory=4, backlog=3, holding_cost=2.0, backlog_cost=5.0
        )
        == 4 * 2.0 + 3 * 5.0
    )


def test_local_cost_estimator_zero_state_zero_cost() -> None:
    assert local_cost_estimator(inventory=0, backlog=0) == 0.0
