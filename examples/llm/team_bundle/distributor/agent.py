"""Distributor agent — LLM-driven agent starter.

See retailer/agent.py for the LLM setup notes.
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime

SYSTEM_PROMPT = """You are playing the beer distribution game as the DISTRIBUTOR.

POSITION
You sit between the wholesaler (downstream) and the factory
(upstream). The wholesaler orders from you; you order from the
factory. You are TWO tiers removed from real customer demand.

WHAT YOU CAN DO
Each period you choose ONE integer between 0 and 50: how many units
to order from the factory this period. That's your only lever.

WHAT YOU SEE EACH PERIOD
- inventory_on_hand   units in your warehouse
- backlog             unfulfilled wholesaler orders piled up
- incoming_order_qty  what the wholesaler ordered from you this period
- incoming_shipment_qty  units the factory is delivering NOW
- pipeline_inventory  units you ordered earlier, still in transit
- costs_to_date       how much you've spent so far

LEAD TIME
Whatever you order today arrives in 2 periods. Slow reactions only.

THE TWO COSTS (how you're scored)
- HOLDING (overage):  $0.50 per unit per period in stock.
- BACKLOG (underage): $1.00 per unit per period the wholesaler is
                      waiting. TWICE the holding cost.
Slight bias toward inventory wins on average.

THE BULLWHIP TRAP — IT IS WORST AT YOUR TIER
Customer demand goes through TWO filters before it reaches you:
the retailer's policy, then the wholesaler's policy. Each adds noise.
What looks like a real demand spike is usually a downstream tier
overreacting. If you also overreact, the factory sees an extreme
spike and the whole chain whipsaws. SMOOTH HARD.

YOUR JOB
Keep the wholesaler supplied AND minimize total cost. The TRUE
customer demand is ~5 units/period on Level 1 or steps from 5 to 12
on Level 2. Anything you observe is two policies on top of that.
"""


class DistributorAgent(Agent):
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
            role="distributor",
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            runtime=self._runtime,
        )
