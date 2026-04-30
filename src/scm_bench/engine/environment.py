"""4-tier supply chain simulation environment.

Strict contract — no cross-tier observation, no scenario-side supply
modifier (factory has unlimited supply), no engine-side max_order clamp
(order budgets are enforced by the validator/harness). The engine takes
a pre-supplied demand sequence and runs a fixed 7-step per-tick pipeline:

    receive shipments → receive orders → fulfill → record history →
    place orders → cost → advance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scm_bench.engine.state import TIER_NAMES, ChainState
from scm_bench.metrics.core import (
    average,
    calculate_bullwhip,
    coefficient_of_variation,
    fill_rate,
)


@dataclass
class TierResult:
    name: str
    total_holding_cost: float
    total_backlog_cost: float
    total_cost: float
    order_cv: float
    avg_inventory: float
    avg_backlog: float
    fill_rate: float


@dataclass
class SimulationResult:
    periods_run: int
    total_cost: float
    bullwhip: float
    chain_fill_rate: float
    average_inventory: float
    average_backlog: float
    tier_results: dict[str, TierResult]
    demand_history: list[int]


@dataclass
class EnvironmentConfig:
    """Engine configuration. Defaults match the v1 reference scenario."""

    horizon: int = 50
    initial_inventory: int = 12
    initial_pipeline: tuple[int, int] = (8, 8)
    initial_orders: tuple[int, int] = (8, 8)
    holding_cost: float = 0.50
    backlog_cost: float = 1.00


class Environment:
    """4-tier supply chain simulation.

    Lifecycle:
        env = Environment(config)
        env.reset()
        for tick in range(config.horizon):
            decisions = {tier: int_order_qty for tier in TIER_NAMES}
            done, info = env.step(consumer_demand=int, decisions=decisions)
            if done:
                break
        result = env.get_result()
    """

    TIER_ORDER = TIER_NAMES

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self._state: ChainState | None = None
        self._inventory_history: dict[str, list[int]] = {t: [] for t in self.TIER_ORDER}
        self._backlog_history: dict[str, list[int]] = {t: [] for t in self.TIER_ORDER}
        self._orders_received_this_period: dict[str, int] = {}

    def reset(self) -> None:
        self._state = ChainState.create_initial(
            initial_inventory=self.config.initial_inventory,
            holding_cost=self.config.holding_cost,
            backlog_cost=self.config.backlog_cost,
        )
        self._inventory_history = {t: [] for t in self.TIER_ORDER}
        self._backlog_history = {t: [] for t in self.TIER_ORDER}
        self._orders_received_this_period = {t: 0 for t in self.TIER_ORDER}

        for tier_name in self.TIER_ORDER:
            tier = self._state.get_tier(tier_name)
            tier.incoming_shipments = list(self.config.initial_pipeline)
            tier.incoming_orders = list(self.config.initial_orders)

    def step(self, consumer_demand: int, decisions: dict[str, int]) -> tuple[bool, dict]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # 1. Receive shipments (from upstream, arrived via shipping lead time).
        self._receive_shipments()

        # 2. Receive orders (retailer gets consumer demand; others get
        #    downstream orders placed in the previous tick).
        self._state.consumer_demand_history.append(consumer_demand)
        self._receive_orders(consumer_demand)

        # 3. Fulfill orders (ship to downstream, accumulate backlog on
        #    shortfall; factory ships from unlimited supply).
        self._fulfill_orders()

        # 4. Record state for history (after fulfillment, before decisions).
        for tier_name in self.TIER_ORDER:
            tier = self._state.get_tier(tier_name)
            self._inventory_history[tier_name].append(tier.inventory)
            self._backlog_history[tier_name].append(tier.backlog)

        # 5. Place orders (decisions feed upstream order queues).
        self._place_orders(decisions)

        # 6. Calculate costs.
        for tier_name in self.TIER_ORDER:
            self._state.get_tier(tier_name).calculate_period_cost()

        # 7. Advance period.
        self._state.advance_period()
        done = self._state.period >= self.config.horizon
        info = {
            "period": self._state.period,
            "total_cost": self._state.total_cost,
            "total_inventory": self._state.total_inventory,
            "total_backlog": self._state.total_backlog,
        }
        return done, info

    def _receive_shipments(self) -> None:
        for tier_name in self.TIER_ORDER:
            tier = self._state.get_tier(tier_name)
            tier.receive_shipment()
            tier.incoming_shipments.append(0)

    def _receive_orders(self, consumer_demand: int) -> None:
        retailer = self._state.get_tier("retailer")
        retailer.incoming_orders[0] = consumer_demand
        self._orders_received_this_period["retailer"] = retailer.receive_order()
        retailer.incoming_orders.append(0)

        for tier_name in self.TIER_ORDER[1:]:
            tier = self._state.get_tier(tier_name)
            self._orders_received_this_period[tier_name] = tier.receive_order()
            tier.incoming_orders.append(0)

    def _fulfill_orders(self) -> None:
        for i, tier_name in enumerate(self.TIER_ORDER):
            tier = self._state.get_tier(tier_name)
            total_to_fill = self._orders_received_this_period[tier_name] + tier.backlog
            shipped = min(tier.inventory, total_to_fill)
            tier.inventory -= shipped
            tier.backlog = total_to_fill - shipped
            tier.shipment_history.append(shipped)

            if i > 0:  # not retailer; ship downstream
                downstream_name = self.TIER_ORDER[i - 1]
                downstream = self._state.get_tier(downstream_name)
                downstream.incoming_shipments[-1] += shipped

    def _place_orders(self, decisions: dict[str, int]) -> None:
        for i, tier_name in enumerate(self.TIER_ORDER[:-1]):
            tier = self._state.get_tier(tier_name)
            order_qty = max(0, int(decisions.get(tier_name, 0)))
            tier.order_history.append(order_qty)

            upstream_name = self.TIER_ORDER[i + 1]
            upstream = self._state.get_tier(upstream_name)
            upstream.incoming_orders[-1] += order_qty

        # Factory orders from unlimited supply.
        factory = self._state.get_tier("factory")
        factory_order = max(0, int(decisions.get("factory", 0)))
        factory.order_history.append(factory_order)
        factory.incoming_shipments[-1] += factory_order

    def get_result(self) -> SimulationResult:
        if self._state is None:
            raise RuntimeError("No simulation has been run")

        tier_results: dict[str, TierResult] = {}
        total_cost = 0.0

        for tier_name in self.TIER_ORDER:
            tier = self._state.get_tier(tier_name)
            t_total = tier.cumulative_holding_cost + tier.cumulative_backlog_cost
            tier_results[tier_name] = TierResult(
                name=tier_name,
                total_holding_cost=tier.cumulative_holding_cost,
                total_backlog_cost=tier.cumulative_backlog_cost,
                total_cost=t_total,
                order_cv=coefficient_of_variation(tier.order_history),
                avg_inventory=average(self._inventory_history[tier_name]),
                avg_backlog=average(self._backlog_history[tier_name]),
                fill_rate=fill_rate(tier.shipment_history, tier.orders_received_history),
            )
            total_cost += t_total

        factory = self._state.get_tier("factory")
        bullwhip = calculate_bullwhip(factory.order_history, self._state.consumer_demand_history)

        return SimulationResult(
            periods_run=self._state.period,
            total_cost=total_cost,
            bullwhip=bullwhip,
            chain_fill_rate=float(np.mean([tr.fill_rate for tr in tier_results.values()])),
            average_inventory=float(np.mean([tr.avg_inventory for tr in tier_results.values()])),
            average_backlog=float(np.mean([tr.avg_backlog for tr in tier_results.values()])),
            tier_results=tier_results,
            demand_history=self._state.consumer_demand_history.copy(),
        )

    # --- read-only inspection (used by harness; does not mutate state) ---

    @property
    def state(self) -> ChainState:
        if self._state is None:
            raise RuntimeError("Must call reset() before inspecting state")
        return self._state

    def orders_received_this_period(self, tier_name: str) -> int:
        return self._orders_received_this_period.get(tier_name, 0)
