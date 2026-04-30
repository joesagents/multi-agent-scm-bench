"""Internal 4-tier supply chain simulation state.

Ported verbatim from scm_bench v1 state.py — these structs anchor
reproducibility. Field names, queue semantics, and cost accumulation are
identical to v1; the engine parity test in tests/test_engine_parity_v1.py
gates any change here.
"""

from dataclasses import dataclass, field

TIER_NAMES: list[str] = ["retailer", "wholesaler", "distributor", "factory"]


@dataclass
class TierState:
    """Internal state for one tier in the supply chain."""

    name: str
    inventory: int = 12
    backlog: int = 0

    incoming_shipments: list[int] = field(default_factory=lambda: [0, 0])
    incoming_orders: list[int] = field(default_factory=lambda: [0, 0])

    order_history: list[int] = field(default_factory=list)
    shipment_history: list[int] = field(default_factory=list)
    orders_received_history: list[int] = field(default_factory=list)

    cumulative_holding_cost: float = 0.0
    cumulative_backlog_cost: float = 0.0

    holding_cost: float = 0.50
    backlog_cost: float = 1.00

    @property
    def cumulative_cost(self) -> float:
        return self.cumulative_holding_cost + self.cumulative_backlog_cost

    def receive_shipment(self) -> int:
        received = self.incoming_shipments.pop(0)
        self.inventory += received
        return received

    def receive_order(self) -> int:
        order = self.incoming_orders.pop(0)
        self.orders_received_history.append(order)
        return order

    def calculate_period_cost(self) -> tuple[float, float]:
        holding = self.inventory * self.holding_cost
        backlog = self.backlog * self.backlog_cost
        self.cumulative_holding_cost += holding
        self.cumulative_backlog_cost += backlog
        return holding, backlog


@dataclass
class ChainState:
    """Complete state for the 4-tier supply chain."""

    period: int = 0
    tiers: dict[str, TierState] = field(default_factory=dict)
    consumer_demand_history: list[int] = field(default_factory=list)

    @classmethod
    def create_initial(
        cls,
        initial_inventory: int = 12,
        holding_cost: float = 0.50,
        backlog_cost: float = 1.00,
    ) -> "ChainState":
        tiers = {
            name: TierState(
                name=name,
                inventory=initial_inventory,
                holding_cost=holding_cost,
                backlog_cost=backlog_cost,
            )
            for name in TIER_NAMES
        }
        return cls(period=0, tiers=tiers, consumer_demand_history=[])

    def get_tier(self, name: str) -> TierState:
        return self.tiers[name]

    def advance_period(self) -> None:
        self.period += 1

    @property
    def total_cost(self) -> float:
        return sum(tier.cumulative_cost for tier in self.tiers.values())

    @property
    def total_inventory(self) -> int:
        return sum(tier.inventory for tier in self.tiers.values())

    @property
    def total_backlog(self) -> int:
        return sum(tier.backlog for tier in self.tiers.values())
