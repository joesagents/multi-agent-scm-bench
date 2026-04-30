"""Factory agent — agent-edited starter.

The factory is the most upstream tier — it has no upstream supplier.
Its `order_qty` represents production starts. There is no `inbox`
upstream of the factory.

Default behavior: produce exactly what the downstream tier asked for.
Replace the body of `step()` with your own decision rule.
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message


class FactoryAgent(Agent):
    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        super().reset(role=role, config=config, seed=seed)

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        # Available on `observation` (LocalObservation):
        #   observation.timestep                   int   current period
        #   observation.role                       str   "factory"
        #   observation.inventory_on_hand          int   units of finished goods in your warehouse
        #   observation.backlog                    int   unfulfilled distributor orders
        #   observation.incoming_order_qty         int   what the distributor ordered THIS period
        #   observation.incoming_shipment_qty      int   units finishing production NOW
        #   observation.pipeline_inventory         int   units in production, not yet finished
        #   observation.order_history_window       list[int]   recent distributor orders (≤ 8)
        #   observation.shipment_history_window    list[int]   recent production completions (≤ 8)
        #   observation.costs_to_date.total        float   $ accumulated cost
        #   observation.costs_to_date.holding      float   $ from holding inventory ($0.50/unit/period)
        #   observation.costs_to_date.backlog      float   $ from backlog ($1.00/unit/period)
        #
        # `order_qty` here means "units to START PRODUCING this period" —
        # the factory has no upstream supplier. Production lead time = 2 periods.
        #
        # Mirror policy: produce whatever the distributor just asked for.
        return AgentDecision(order_qty=observation.incoming_order_qty)
