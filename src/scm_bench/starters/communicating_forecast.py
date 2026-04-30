"""Communicating Forecast Agent — base-stock + emit a forecast message upstream.

Teaches the A2A-style messaging interface. Each tick, the agent computes
its own short-horizon forecast (moving average of incoming orders) and
sends it upstream as a `forecast` message.

Phase 1: messages are recorded but not delivered (the bus is a no-op).
Phase 2: messages will be delivered to the upstream tier's inbox subject
to the scenario's communication graph and delay/drop policy.
"""

from __future__ import annotations

from typing import Any

from scm_bench.memory.store import BoundedBufferMemory
from scm_bench.messaging.envelope import Message
from scm_bench.messaging.types import MessageType
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation
from scm_bench.starters.base_stock import BaseStockAgent

UPSTREAM_OF: dict[str, str | None] = {
    "retailer": "wholesaler",
    "wholesaler": "distributor",
    "distributor": "factory",
    "factory": None,  # factory has no upstream
}


class CommunicatingForecastAgent(Agent):
    DEFAULT_TARGET = 24
    DEFAULT_WINDOW = 4
    DEFAULT_FORECAST_HORIZON = 4

    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        super().reset(role=role, config=config, seed=seed)
        self._target = int(config.get("target_stock", self.DEFAULT_TARGET))
        self._window = int(config.get("window", self.DEFAULT_WINDOW))
        self._horizon = int(config.get("forecast_horizon", self.DEFAULT_FORECAST_HORIZON))
        self._mem = BoundedBufferMemory(max_entries=self._window)
        # Reuse base-stock policy as the order rule
        self._policy = BaseStockAgent()
        self._policy.reset(role=role, config={"target_stock": self._target}, seed=seed)

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        self._mem.append(observation.incoming_order_qty)
        history = self._mem.snapshot()
        forecast_value = sum(history) / len(history) if history else 0.0
        decision = self._policy.step(observation, inbox, t)

        upstream = UPSTREAM_OF[self.role]
        messages: list[Message] = []
        if upstream is not None:
            messages.append(
                Message(
                    receiver_role=upstream,
                    type=MessageType.FORECAST,
                    payload={
                        "forecast_horizon": self._horizon,
                        "forecast_values": [round(forecast_value)] * self._horizon,
                    },
                )
            )

        return AgentDecision(order_qty=decision.order_qty, messages=messages)
