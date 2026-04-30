"""Message envelope — the wire format for inter-agent communication."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scm_bench.messaging.types import VALID_ROLES, MessageType


class Message(BaseModel):
    """Outgoing message from an agent.

    Agents construct these in their AgentDecision; the harness assigns
    `message_id` and `ts_sent`/`ts_delivered` when dispatching through the
    bus. In Phase 1 the bus is a no-op (messages are recorded but not
    delivered); Phase 2 wires the typed delivery semantics.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts_sent: int | None = None
    ts_delivered: int | None = None
    sender_role: str | None = None  # filled by harness from the calling agent's role
    receiver_role: str
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        if self.receiver_role not in VALID_ROLES:
            raise ValueError(
                f"Message.receiver_role must be one of {sorted(VALID_ROLES)}, "
                f"got {self.receiver_role!r}"
            )
