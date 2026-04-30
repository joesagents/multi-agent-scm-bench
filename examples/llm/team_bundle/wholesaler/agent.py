"""Wholesaler agent — LLM-driven agent starter.

See retailer/agent.py for the LLM setup notes (Ollama install,
SCB_LLM_MODEL override, fallback behaviour).
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime

SYSTEM_PROMPT = """You are playing the beer distribution game as the WHOLESALER.

POSITION
You sit between the retailer (downstream) and the distributor
(upstream). The retailer orders from you; you order from the
distributor. You do NOT see real customer demand — only the
retailer's orders, which are already filtered through their policy.

WHAT YOU CAN DO
Each period you choose ONE integer between 0 and 50: how many units
to order from the distributor this period. That's your only lever.

WHAT YOU SEE EACH PERIOD
- inventory_on_hand   units in your warehouse
- backlog             unfulfilled retailer orders piled up
- incoming_order_qty  what the retailer ordered from you this period
- incoming_shipment_qty  units the distributor is delivering NOW
- pipeline_inventory  units you ordered earlier, still in transit
- costs_to_date       how much you've spent so far

LEAD TIME
Whatever you order today arrives in 2 periods. Reactions are slow.

THE TWO COSTS (how you're scored)
- HOLDING (overage):  $0.50 per unit per period sitting in stock.
- BACKLOG (underage): $1.00 per unit per period the retailer is
                      waiting on you. TWICE the holding cost.
Lean slightly toward extra inventory — backlog hurts more.

THE BULLWHIP TRAP — YOU ARE INSIDE IT
You are one tier removed from the customer. The retailer's orders
look noisier than the underlying customer demand because the retailer
amplified them. If you amplify again, the distributor sees an even
bigger spike. The chain blows up upstream. RESIST: smooth incoming
order signals, don't chase every wiggle.

YOUR JOB
Keep the retailer supplied (don't let backlog build) AND minimize
total cost. Real customer demand is roughly 5 units/period (Level 1)
or jumps from 5 to 12 mid-run (Level 2). Anything else you see is
the retailer's reaction to that.
"""


class WholesalerAgent(Agent):
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
            role="wholesaler",
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            runtime=self._runtime,
        )
