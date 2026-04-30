"""Retailer agent — LLM-driven agent starter.

Each tick this agent asks a local LLM (default: Ollama, gemma4:e4b)
for an order quantity and returns it. If the LLM call fails (Ollama
not running, network down) the helper falls back to the mirror policy
so the bundle still validates and runs.

Edit two things:
- `SYSTEM_PROMPT` below — the brief sent to the model. Make it sharper,
  add the policy you want the model to follow, give it constraints.
- The body of `step()` — if you want anything more than "ask the model
  what to order this period," do it here.

Run requirements (only if you want real LLM behaviour):
- Ollama installed:  https://ollama.com
- A pulled model:    `ollama pull gemma4:e4b`   (~9 GB, M-series friendly)
- Override model:    `export SCB_LLM_MODEL=gemma4:e2b`
- Override backend:  `export SCB_LLM_BACKEND=openai_compat`
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime

SYSTEM_PROMPT = """You are playing the beer distribution game as the RETAILER.

POSITION
You are at the bottom of a 4-tier chain. Customers order from you;
you order from the WHOLESALER upstream. You are the only tier that
sees real customer demand.

WHAT YOU CAN DO
Each period you choose ONE integer between 0 and 50: how many units
to order from the wholesaler this period. That's your only lever.

WHAT YOU SEE EACH PERIOD
- inventory_on_hand   units sitting in your warehouse right now
- backlog             unfulfilled customer orders piled up
- incoming_order_qty  customer demand THIS period (your real signal)
- incoming_shipment_qty  units the wholesaler is delivering NOW
- pipeline_inventory  units you ordered earlier, still in transit
- costs_to_date       how much you've spent so far

LEAD TIME
Whatever you order today arrives in 2 periods. Anything you order now
to fix today's stockout is too late for today — it lands the period
after next.

THE TWO COSTS (this is how you're scored)
- HOLDING (overage):  $0.50 per unit you hold per period.
                      Order too much → inventory piles up → you bleed
                      $0.50/unit every tick it sits there.
- BACKLOG (underage): $1.00 per unit you fail to deliver per period.
                      Order too little → customers wait → you bleed
                      $1.00/unit every tick they're unhappy.
Backlog costs TWICE as much as holding. Erring slightly on the high
side is cheaper than erring on the low side.

THE BULLWHIP TRAP
If you over-react to one period of high demand and order a lot, the
wholesaler sees a spike and over-orders from the distributor, who
over-orders from the factory. By the time excess inventory cascades
back, customer demand may already be normal again. Smooth your orders.

YOUR JOB
Keep your store alive (don't stockout for long) AND minimize total
cost. The customer demand on Level 1 is roughly stable around 5
units/period; on Level 2 it jumps from ~5 to ~12 mid-run.
"""


class RetailerAgent(Agent):
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
            role="retailer",
            system_prompt=SYSTEM_PROMPT,
            observation=observation,
            runtime=self._runtime,
        )
