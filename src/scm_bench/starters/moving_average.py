"""Moving Average Agent — order the mean of the last K incoming orders.

Teaches local memory: the agent must keep state across ticks because the
observation window is bounded.
"""

from __future__ import annotations

from typing import Any

from scm_bench.memory.store import BoundedBufferMemory
from scm_bench.messaging.envelope import Message
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation


class MovingAverageAgent(Agent):
    DEFAULT_WINDOW = 4

    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        super().reset(role=role, config=config, seed=seed)
        self._window = int(config.get("window", self.DEFAULT_WINDOW))
        self._mem = BoundedBufferMemory(max_entries=self._window)

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        self._mem.append(observation.incoming_order_qty)
        history = self._mem.snapshot()
        avg = sum(history) / len(history) if history else 0.0
        return AgentDecision(order_qty=max(0, round(avg)))
