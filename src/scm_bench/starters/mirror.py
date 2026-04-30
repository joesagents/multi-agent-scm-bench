"""Mirror Agent — order what you just received.

Teaches contract compliance and is the canonical bullwhip baseline.
Formula: order_t = incoming_order_qty
"""

from __future__ import annotations

from scm_bench.messaging.envelope import Message
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation


class MirrorAgent(Agent):
    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        return AgentDecision(order_qty=observation.incoming_order_qty)
