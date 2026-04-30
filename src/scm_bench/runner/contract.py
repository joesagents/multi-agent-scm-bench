"""Per-tick manifest contract enforcement.

The agent manifest declares what each agent is allowed to do
(supports_messages, supports_tools) and how much state it may hold
(memory_max_entries). Without enforcement these fields would be
advisory only, which makes for unfair scoring: a submission could
declare a tight contract and then quietly violate it.

The harness calls these helpers each tick. A violation is fatal: the
run aborts with `E_CONTRACT_VIOLATION` and the offending role/reason
goes back to the validator / batch-run summary so the aggregate can
see it before scoring.

What is enforced today:
- supports_messages: every emitted Message.type must be in the
  declared list.
- memory_max_entries: if the agent exposes `memory` with a `snapshot()`
  method (the documented convention; see `scm_bench.memory`),
  the snapshot length must stay within the declared cap.

What is NOT yet enforced:
- supports_tools: there is no tool-call surface in `AgentDecision`
  yet, so there is nothing to check. When tools land, the same
  pattern applies.
- Memory state held in opaque attributes (e.g. raw `self.history`
  lists) is invisible to the auditor. Documented in the SDK README as
  the reason agents should use `make_memory()`.
"""

from __future__ import annotations

from scm_bench.messaging.envelope import Message
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.manifest import AgentManifest


class ContractViolationError(Exception):
    """Raised when an agent step violates its declared manifest contract."""

    code = "E_CONTRACT_VIOLATION"

    def __init__(self, role: str, reason: str, *, t: int) -> None:
        super().__init__(f"role={role!r} t={t}: {reason}")
        self.role = role
        self.reason = reason
        self.t = t


def check_messages(
    *, role: str, manifest: AgentManifest, messages: list[Message], t: int
) -> None:
    allowed = set(manifest.supports_messages)
    for msg in messages:
        if msg.type not in allowed:
            raise ContractViolationError(
                role,
                f"emitted message of type {msg.type.value!r} but manifest "
                f"declares supports_messages={[m.value for m in manifest.supports_messages]}",
                t=t,
            )


def check_memory(
    *, role: str, manifest: AgentManifest, agent: Agent, t: int
) -> None:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return
    snapshot = getattr(memory, "snapshot", None)
    if snapshot is None or not callable(snapshot):
        return
    try:
        size = len(snapshot())
    except Exception:
        # Memory implementations that crash on snapshot are themselves a bug,
        # but we don't want to mask the underlying agent error here — let the
        # next tick surface it through the normal error path.
        return
    if size > manifest.memory_max_entries:
        raise ContractViolationError(
            role,
            f"memory snapshot has {size} entries but manifest declares "
            f"memory_max_entries={manifest.memory_max_entries}",
            t=t,
        )


__all__ = ["ContractViolationError", "check_memory", "check_messages"]
