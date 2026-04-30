"""Named scenarios used in the course.

`intro_step_demand` is the validator's smoke scenario.

`s1_1` and `s2_3` are the public benchmark levels:

- **S1.1 — Stable** (Level 1, viability): low-noise stationary demand,
  365-period horizon. Pass criterion: completes without error.
- **S2.3 — Demand shock** (Level 2, performance): step jump mid-horizon
  (sudden regime change). Tests whether agents detect and adapt.

(Level 3 — hidden disruption — is not part of this public build.
The hook is preserved so an operator or researcher can plug in a
custom mechanism without touching the engine.)
"""

from __future__ import annotations

import random

from scm_bench.engine.environment import EnvironmentConfig
from scm_bench.scenarios.library import iid_uniform, step
from scm_bench.scenarios.scenario import DemandFn, Scenario

INTRO_STEP_DEMAND = Scenario(
    id="intro_step_demand",
    name="Intro: Step Demand",
    description=(
        "Classic step demand. Demand is 4 for the first 5 periods, then "
        "jumps to 8. Used as the smoke scenario for bundle validation."
    ),
    demand_fn=step(low=4, high=8, switch_at=5),
    env_config=EnvironmentConfig(
        horizon=30,
        initial_inventory=12,
        initial_pipeline=(4, 4),
        initial_orders=(4, 4),
        holding_cost=0.50,
        backlog_cost=1.00,
    ),
    observation_window=8,
)


S1_1_STABLE = Scenario(
    id="s1.1",
    name="Level 1 — Stable",
    description=(
        "Stationary low-noise demand for 365 periods. Viability test: "
        "agents must complete the run without error."
    ),
    demand_fn=iid_uniform(low=3, high=7),
    env_config=EnvironmentConfig(
        horizon=365,
        initial_inventory=12,
        initial_pipeline=(5, 5),
        initial_orders=(5, 5),
        holding_cost=0.50,
        backlog_cost=1.00,
    ),
    observation_window=8,
)


S2_3_DEMAND_SHOCK = Scenario(
    id="s2.3",
    name="Level 2 — Demand Shock",
    description=(
        "Stable demand of ~5 units for 180 periods, then a sustained "
        "step to ~12 for the remaining 185 periods. Tests whether agents "
        "detect regime change and adapt without amplifying the bullwhip."
    ),
    demand_fn=step(low=5, high=12, switch_at=180),
    env_config=EnvironmentConfig(
        horizon=365,
        initial_inventory=12,
        initial_pipeline=(5, 5),
        initial_orders=(5, 5),
        holding_cost=0.50,
        backlog_cost=1.00,
    ),
    observation_window=12,
)




SCENARIOS: dict[str, Scenario] = {
    INTRO_STEP_DEMAND.id: INTRO_STEP_DEMAND,
    S1_1_STABLE.id: S1_1_STABLE,
    S2_3_DEMAND_SHOCK.id: S2_3_DEMAND_SHOCK,
}


PUBLIC_LEVELS: tuple[str, str] = ("s1.1", "s2.3")


def get(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario: {scenario_id!r}. Available: {sorted(SCENARIOS)}"
        )
    return SCENARIOS[scenario_id]


def list_ids() -> list[str]:
    return sorted(SCENARIOS)
