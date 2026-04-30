"""Built-in local tools available to agent agents."""

from __future__ import annotations

from collections.abc import Sequence

from scm_bench.tools.registry import ToolRegistry


def forecast_moving_average(values: Sequence[int | float], window: int = 4) -> float:
    if not values:
        return 0.0
    window = max(1, min(window, len(values)))
    tail = values[-window:]
    return sum(tail) / window


def forecast_exponential_smoothing(values: Sequence[int | float], alpha: float = 0.3) -> float:
    if not values:
        return 0.0
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    s = float(values[0])
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s


def local_cost_estimator(
    inventory: int,
    backlog: int,
    holding_cost: float = 0.50,
    backlog_cost: float = 1.00,
) -> float:
    return inventory * holding_cost + backlog * backlog_cost


BUILTIN_TOOLS: dict[str, object] = {
    "forecast_moving_average": forecast_moving_average,
    "forecast_exponential_smoothing": forecast_exponential_smoothing,
    "local_cost_estimator": local_cost_estimator,
}


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name, fn in BUILTIN_TOOLS.items():
        reg.register(name, fn)  # type: ignore[arg-type]
    return reg
