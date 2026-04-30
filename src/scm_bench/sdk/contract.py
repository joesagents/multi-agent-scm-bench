"""SDK contract — what agents see and return.

These are the exact types that cross the agent boundary. They are
deliberately strict (Pydantic `extra="forbid"`) so any malformed return
from a submitted agent is caught at the contract surface, not deep inside
the engine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scm_bench.messaging.envelope import Message
from scm_bench.messaging.types import VALID_ROLES


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holding: float = 0.0
    stockout: float = 0.0
    total: float = 0.0


class LocalObservation(BaseModel):
    """What the harness hands the agent each tick.

    Strictly local: no full-chain visibility, no future demand, no
    cross-agent state. The harness builds this from the engine's TierState
    and the agent's role.
    """

    model_config = ConfigDict(extra="forbid")

    timestep: int = Field(ge=0)
    role: str
    inventory_on_hand: int = Field(ge=0)
    backlog: int = Field(ge=0)
    incoming_order_qty: int = Field(ge=0)
    incoming_shipment_qty: int = Field(ge=0)
    pipeline_inventory: int = Field(ge=0)
    order_history_window: list[int] = Field(default_factory=list)
    shipment_history_window: list[int] = Field(default_factory=list)
    costs_to_date: CostBreakdown = Field(default_factory=CostBreakdown)

    @field_validator("role")
    @classmethod
    def _role_must_be_valid(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}, got {v!r}")
        return v


class ToolCall(BaseModel):
    """An agent's request to invoke a registered local tool.

    Phase 1 records tool calls but does not enforce the declaration check;
    Phase 2 will reject undeclared tools.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    """The single object an agent returns per tick."""

    model_config = ConfigDict(extra="forbid")

    order_qty: int = Field(ge=0)
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens_used: int = Field(default=0, ge=0)
