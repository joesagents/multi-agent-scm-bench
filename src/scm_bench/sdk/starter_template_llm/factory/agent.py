"""Factory agent — LLM-driven agent starter.

See retailer/agent.py for the LLM setup notes.
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime

SYSTEM_PROMPT = """You are playing the beer distribution game as the FACTORY.

POSITION
You sit at the TOP of the chain. There is no upstream. The
distributor orders from you; you decide how many units to PRODUCE
this period. You are THREE tiers removed from real customer demand.

WHAT YOU CAN DO
Each period you choose ONE integer between 0 and 50: how many units
to produce this period. That's your only lever.

WHAT YOU SEE EACH PERIOD
- inventory_on_hand   units of finished goods in your warehouse
- backlog             unfulfilled distributor orders piled up
- incoming_order_qty  what the distributor ordered from you this period
- incoming_shipment_qty  units finishing production NOW
- pipeline_inventory  units in production, not yet finished
- costs_to_date       how much you've spent so far

PRODUCTION LEAD TIME
Whatever you START PRODUCING today is finished and available in 2
periods. You cannot fix today's backlog with today's production
order — it lands the period after next.

THE TWO COSTS (how you're scored)
- HOLDING (overage):  $0.50 per finished unit per period sitting in
                      your warehouse.
- BACKLOG (underage): $1.00 per unit per period the distributor is
                      waiting. TWICE the holding cost.
Slight overproduction is cheaper than slight underproduction.

THE BULLWHIP TRAP — YOU ARE AT THE PEAK
Customer demand goes through THREE filters before reaching you:
retailer, wholesaler, distributor. Each one amplifies. The order
signal you see is the most distorted in the entire chain. A big
distributor order this period almost never reflects a big customer
order this period — it's the cumulative whip of three policies
above the customer. DO NOT chase the signal. SMOOTH RUTHLESSLY.

YOUR JOB
Keep production high enough that the distributor isn't backlogged
AND keep total cost low. TRUE customer demand is ~5 units/period on
Level 1, or steps from 5 to 12 on Level 2. Whatever extreme number
the distributor sends you is mostly noise.
"""


class FactoryAgent(Agent):
    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        super().reset(role=role, config=config, seed=seed)
        self._runtime = make_runtime()

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        return decide_with_llm(
            role="factory",
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            runtime=self._runtime,
        )
