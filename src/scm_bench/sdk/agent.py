"""Agent base class — what agents subclass.

The contract is intentionally tiny: implement step(); optionally override
reset() and on_feedback(). Local memory is the agent's responsibility;
make_memory() in scm_bench.memory is provided as a default
quota-respecting implementation but agents may use plain Python state if
they prefer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scm_bench.messaging.envelope import Message
from scm_bench.sdk.contract import AgentDecision, LocalObservation


class Agent(ABC):
    """Abstract base for all submitted agents.

    Lifecycle (called by the harness, not by user code):
        agent = AgentClass()
        agent.reset(role="retailer", config={...}, seed=42)
        for t in range(horizon):
            decision = agent.step(obs, inbox, t)
        agent.on_feedback(feedback_dict)  # optional, end-of-run
    """

    role: str = ""
    config: dict[str, Any] = {}

    def reset(self, *, role: str, config: dict[str, Any], seed: int) -> None:
        """Initialize per-run state. Subclasses may override but must call super()."""
        self.role = role
        self.config = dict(config)

    @abstractmethod
    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision: ...

    def on_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: B027
        """Optional end-of-run feedback hook. Default no-op."""
        return None
