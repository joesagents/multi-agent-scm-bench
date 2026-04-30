"""Retailer agent — agent-edited starter.

This file is the *only* place you control the retailer's policy. The
harness instantiates `RetailerAgent` once per run and calls `step()`
once per simulated period.

Default behavior: order exactly what the customer asked for. This is
the canonical "Mirror" policy and it produces the textbook bullwhip.
Replace the body of `step()` with your own decision rule.
"""

from __future__ import annotations

from typing import Any

from scm_bench.sdk import Agent, AgentDecision, LocalObservation, Message


class RetailerAgent(Agent):
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
        #   observation.role                       str   "retailer"
        #   observation.inventory_on_hand          int   units in your store
        #   observation.backlog                    int   unfulfilled customer orders
        #   observation.incoming_order_qty         int   customer demand THIS period
        #   observation.incoming_shipment_qty      int   units arriving from wholesaler NOW
        #   observation.pipeline_inventory         int   units ordered earlier, still in transit
        #   observation.order_history_window       list[int]   recent customer demand (≤ 8)
        #   observation.shipment_history_window    list[int]   recent deliveries from upstream (≤ 8)
        #   observation.costs_to_date.total        float   $ accumulated cost
        #   observation.costs_to_date.holding      float   $ from holding inventory ($0.50/unit/period)
        #   observation.costs_to_date.backlog      float   $ from backlog ($1.00/unit/period)
        #
        # Mirror policy: order whatever the customer just asked for.
        return AgentDecision(order_qty=observation.incoming_order_qty)
