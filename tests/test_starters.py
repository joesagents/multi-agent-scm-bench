"""Reference starter agents — each runs end-to-end on intro_step_demand.

Confirms the four reference policies in `scm_bench.starters`
satisfy the contract and produce finite, sensible costs. They are the
ground-truth examples agents read while implementing their own
policies.
"""

from __future__ import annotations

import pytest

from scm_bench.runner.harness import run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.sdk.agent import Agent
from scm_bench.starters.base_stock import BaseStockAgent
from scm_bench.starters.communicating_forecast import (
    CommunicatingForecastAgent,
)
from scm_bench.starters.mirror import MirrorAgent
from scm_bench.starters.moving_average import MovingAverageAgent

STARTER_FACTORIES = {
    "mirror": MirrorAgent,
    "moving_average": MovingAverageAgent,
    "base_stock": BaseStockAgent,
    "communicating_forecast": CommunicatingForecastAgent,
}


@pytest.mark.parametrize("name,factory", list(STARTER_FACTORIES.items()))
def test_starter_runs_end_to_end(name: str, factory: type[Agent]) -> None:
    scenario = scenarios_builtin.INTRO_STEP_DEMAND
    agents = {role: factory() for role in ("retailer", "wholesaler", "distributor", "factory")}
    record = run_one(
        agents=agents,
        scenario=scenario,
        seed=0,
        run_id=f"smoke-{name}",
    )
    assert record.result is not None
    assert record.result.periods_run == scenario.env_config.horizon
    assert record.result.total_cost >= 0.0
    assert 0.0 <= record.result.chain_fill_rate <= 1.0
