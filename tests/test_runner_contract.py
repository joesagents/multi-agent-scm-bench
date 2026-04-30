"""Manifest contract enforcement.

Pins the per-tick checks the harness now runs when an `agent_manifests`
dict is supplied: emitted message types must be in supports_messages,
and exposed memory must stay within memory_max_entries.
"""

from __future__ import annotations

import pytest

from scm_bench.messaging.envelope import Message
from scm_bench.messaging.types import MessageType
from scm_bench.runner.contract import (
    ContractViolationError,
    check_memory,
    check_messages,
)
from scm_bench.runner.harness import run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision
from scm_bench.sdk.manifest import AgentManifest


def _manifest(
    role: str,
    *,
    supports_messages: list[MessageType] | None = None,
    memory_max_entries: int = 32,
    memory_mode: str = "stateless",
) -> AgentManifest:
    from scm_bench import SDK_VERSION

    return AgentManifest(
        role=role,
        agent_name=f"{role}-test",
        entrypoint="agent.py:Agent",
        memory_mode=memory_mode,  # type: ignore[arg-type]
        memory_max_entries=memory_max_entries,
        supports_tools=[],
        supports_messages=supports_messages or [],
        sdk_version=SDK_VERSION,
    )


class _NoiseAgent(Agent):
    """Always emits a FORECAST message regardless of declaration."""

    def step(self, observation, inbox, t):
        return AgentDecision(
            order_qty=0,
            messages=[
                Message(receiver_role="wholesaler", type=MessageType.FORECAST)
            ],
        )


class _GreedyMemoryAgent(Agent):
    """Hoards a list `self.memory.snapshot()` returns longer than declared."""

    class _Memory:
        def __init__(self):
            self._items = []

        def snapshot(self):
            return list(self._items)

        def append(self, item):
            self._items.append(item)

    def reset(self, *, role, config, seed):
        super().reset(role=role, config=config, seed=seed)
        self.memory = self._Memory()

    def step(self, observation, inbox, t):
        self.memory.append(t)  # grows by 1 every tick
        return AgentDecision(order_qty=0)


class _SilentAgent(Agent):
    def step(self, observation, inbox, t):
        return AgentDecision(order_qty=0)


def test_check_messages_passes_when_declared() -> None:
    m = _manifest("retailer", supports_messages=[MessageType.FORECAST])
    check_messages(
        role="retailer",
        manifest=m,
        messages=[Message(receiver_role="wholesaler", type=MessageType.FORECAST)],
        t=0,
    )


def test_check_messages_rejects_undeclared() -> None:
    m = _manifest("retailer", supports_messages=[MessageType.STATUS])
    with pytest.raises(ContractViolationError) as exc:
        check_messages(
            role="retailer",
            manifest=m,
            messages=[
                Message(receiver_role="wholesaler", type=MessageType.FORECAST)
            ],
            t=3,
        )
    assert exc.value.code == "E_CONTRACT_VIOLATION"
    assert exc.value.role == "retailer"
    assert exc.value.t == 3
    assert "forecast" in str(exc.value)


def test_check_memory_passes_within_cap() -> None:
    m = _manifest("retailer", memory_max_entries=4)
    agent = _GreedyMemoryAgent()
    agent.reset(role="retailer", config={}, seed=0)
    for t in range(4):
        agent.memory.append(t)
    check_memory(role="retailer", manifest=m, agent=agent, t=4)


def test_check_memory_rejects_overgrowth() -> None:
    m = _manifest("retailer", memory_max_entries=2)
    agent = _GreedyMemoryAgent()
    agent.reset(role="retailer", config={}, seed=0)
    for t in range(5):
        agent.memory.append(t)
    with pytest.raises(ContractViolationError) as exc:
        check_memory(role="retailer", manifest=m, agent=agent, t=5)
    assert exc.value.code == "E_CONTRACT_VIOLATION"
    assert "memory_max_entries=2" in str(exc.value)


def test_check_memory_silent_when_no_memory_attr() -> None:
    """An agent that doesn't expose `memory` is invisible to the audit
    (documented limitation). Should not raise."""
    m = _manifest("retailer", memory_max_entries=1)
    check_memory(role="retailer", manifest=m, agent=_SilentAgent(), t=0)


def test_run_one_aborts_on_undeclared_message() -> None:
    """End-to-end: undeclared message during a run aborts with the
    contract-violation error code."""
    agents = {
        "retailer": _NoiseAgent(),
        "wholesaler": _SilentAgent(),
        "distributor": _SilentAgent(),
        "factory": _SilentAgent(),
    }
    # retailer declares STATUS but emits FORECAST
    manifests = {
        "retailer": _manifest("retailer", supports_messages=[MessageType.STATUS]),
        "wholesaler": _manifest("wholesaler"),
        "distributor": _manifest("distributor"),
        "factory": _manifest("factory"),
    }
    with pytest.raises(ContractViolationError) as exc:
        run_one(
            agents=agents,
            scenario=scenarios_builtin.INTRO_STEP_DEMAND,
            seed=0,
            run_id="contract-violation-smoke",
            max_ticks=3,
            agent_manifests=manifests,
        )
    assert exc.value.role == "retailer"


def test_run_one_passes_when_no_manifests_supplied() -> None:
    """Built-in starters that don't declare a manifest must still run."""
    agents = {
        "retailer": _NoiseAgent(),  # would violate any manifest
        "wholesaler": _SilentAgent(),
        "distributor": _SilentAgent(),
        "factory": _SilentAgent(),
    }
    record = run_one(
        agents=agents,
        scenario=scenarios_builtin.INTRO_STEP_DEMAND,
        seed=0,
        run_id="no-manifest-smoke",
        max_ticks=3,
        agent_manifests=None,
    )
    assert record.ticks
