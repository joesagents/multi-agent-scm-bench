"""Scenario type — single dataclass replaces v1's 10 scenario subclasses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from scm_bench.engine.environment import EnvironmentConfig

# A demand function is called as demand_fn(period, seed) -> int.
DemandFn = Callable[[int, int], int]


@dataclass(frozen=True)
class Scenario:
    """A named benchmark scenario.

    The Scenario carries:
    - the demand function (deterministic given seed)
    - the engine configuration (horizon, costs, initial state)
    - the observation window (how many recent orders/shipments the agent sees)
    - flags for which messaging features are enabled (Phase 2 wires these)
    """

    id: str
    name: str
    description: str
    demand_fn: DemandFn
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    observation_window: int = 8
    # Phase 2: messaging policy lives here.

    @property
    def horizon(self) -> int:
        return self.env_config.horizon
