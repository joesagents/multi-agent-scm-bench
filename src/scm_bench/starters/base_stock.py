"""Base Stock Agent — close the inventory-position gap to a target.

Teaches local control policy:
    inventory_position = inventory_on_hand + pipeline_inventory + incoming_shipment_qty - backlog
    order_t = max(0, target_stock - inventory_position)

Two optional modes upgrade the textbook policy without changing the
default:

* `demand_driven=True` — replace the fixed `target_stock` with
  `mean_recent_demand × (lead_time + 1)` (the Sterman-textbook
  optimum for zero-uncertainty steady demand). The mean is taken over
  the last `demand_window` ticks of incoming orders. Until the window
  has any data, falls back to `target_stock`.
* `horizon_aware=True` — halt new orders inside the lead-time tail at
  `T − lead_time − 1`, so pipeline stock that would arrive after the
  simulation ends does not inflate holding cost. Requires
  `horizon_hint` in config.

Both modes off (the default) preserve the original fixed-target
behaviour exactly.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from scm_bench.messaging.envelope import Message
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation


class BaseStockAgent(Agent):
    DEFAULT_TARGET = 24
    DEFAULT_LEAD_TIME = 4
    DEFAULT_HORIZON_HINT = 30
    DEFAULT_DEMAND_WINDOW = 12

    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        super().reset(role=role, config=config, seed=seed)
        self._target = int(config.get("target_stock", self.DEFAULT_TARGET))
        self._demand_driven = bool(config.get("demand_driven", False))
        self._horizon_aware = bool(config.get("horizon_aware", False))
        self._lead_time = int(config.get("lead_time", self.DEFAULT_LEAD_TIME))
        self._horizon = int(config.get("horizon_hint", self.DEFAULT_HORIZON_HINT))
        window = int(config.get("demand_window", self.DEFAULT_DEMAND_WINDOW))
        self._demand: deque[int] = deque(maxlen=window)

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        self._demand.append(observation.incoming_order_qty)

        if self._horizon_aware:
            periods_remaining = max(0, self._horizon - observation.timestep)
            if periods_remaining <= self._lead_time + 1:
                return AgentDecision(order_qty=0)

        if self._demand_driven and self._demand:
            mean_demand = sum(self._demand) / len(self._demand)
            target = int(round(mean_demand * (self._lead_time + 1)))
        else:
            target = self._target

        position = (
            observation.inventory_on_hand
            + observation.incoming_shipment_qty
            + observation.pipeline_inventory
            - observation.backlog
        )
        order = max(0, target - position)
        return AgentDecision(order_qty=order)
