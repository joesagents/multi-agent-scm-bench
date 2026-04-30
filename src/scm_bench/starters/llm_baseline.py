"""LLM baseline team — frontier-model anchor for the aggregate summary.

Wraps a cluster-resident Gemma model (default `google/gemma-4-26B-A4B-it`)
as a four-tier team. The four agents share one `LLMRuntime` so the
per-tick `model.generate` runs once over a 4-prompt batch.

Wire-up:

- `make_llm_baseline_team(runtime)` returns `(agents, pre_tick_callback)`.
- The agents themselves are placeholders: their `step()` is never
  called when the harness has the callback wired (the callback returns
  a decision for every role).
- The callback takes `(observations, t)` from the harness, builds one
  prompt per role, calls `runtime.batch_decide`, parses each output
  into an `AgentDecision`, and returns role → AgentDecision.

Why both an agent and a callback: the harness contract still requires
a four-key `agents` dict (the validator's manifest enforcement and the
RunRecord ergonomics depend on it), but the actual decision happens
in the batch fan-out. If batching is later disabled, the agent's
`step()` is a complete fallback path that calls the runtime per-tier.
"""

from __future__ import annotations

import json
import re

from scm_bench.messaging.envelope import Message
from scm_bench.runner.harness import PreTickCallback
from scm_bench.runner.llm_runtime import GenerationResult, LLMRuntime
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation

DEFAULT_MAX_ORDER = 50
DEFAULT_FALLBACK = 8

SYSTEM_PROMPT = (
    "You are a supply chain inventory agent. Minimize cost.\n"
    "Holding: $0.50/unit/period. Backlog: $1.00/unit/period.\n"
    'Reply with ONLY JSON: {"order": <int 0-50>, "reasoning": "<one sentence>"}'
)


def format_prompt(obs: LocalObservation) -> str:
    """Build the per-tick prompt for one role.

    Per-tick prompt builder; the format is fixed to keep token counts
    and decisions comparable across runs.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Period:{obs.timestep} Tier:{obs.role} "
        f"OnHand:{obs.inventory_on_hand} Backlog:{obs.backlog} "
        f"OrdersIn:{obs.incoming_order_qty} "
        f"InTransit:{obs.pipeline_inventory + obs.incoming_shipment_qty} "
        f"Cost:${obs.costs_to_date.total:.0f} MaxOrder:{DEFAULT_MAX_ORDER}"
    )


def parse_order(
    text: str,
    *,
    max_order: int = DEFAULT_MAX_ORDER,
    fallback: int = DEFAULT_FALLBACK,
) -> int:
    """Extract an integer order quantity from the model output.

    Same precedence as v1: JSON block first, then any integer in the
    text, then the fallback (typically `incoming_order_qty` so a parse
    miss degrades to the mirror policy rather than tanking the run).
    """
    try:
        m = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return max(0, min(int(data["order"]), max_order))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    nums = re.findall(r"\b(\d+)\b", text)
    if nums:
        return max(0, min(int(nums[0]), max_order))
    return max(0, min(fallback, max_order))


class LLMBaselineAgent(Agent):
    """Per-tier wrapper around a shared `LLMRuntime`.

    `step()` is the per-tier fallback; the normal path is the team's
    `pre_tick_callback`, which precomputes all four decisions in one
    batched generate. Keeping `step()` correct means the agent still
    works if a caller chooses not to install the callback.
    """

    def __init__(self, runtime: LLMRuntime) -> None:
        self._runtime = runtime

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        prompt = format_prompt(observation)
        out = self._runtime.batch_decide({observation.role: prompt})[observation.role]
        order = parse_order(
            out.text,
            fallback=observation.incoming_order_qty or DEFAULT_FALLBACK,
        )
        return AgentDecision(order_qty=order, tokens_used=out.tokens_used)


def make_llm_pre_tick_callback(runtime: LLMRuntime) -> PreTickCallback:
    """Build a harness `pre_tick_callback` that batches all 4 tiers."""

    def _callback(
        observations: dict[str, LocalObservation], t: int
    ) -> dict[str, AgentDecision]:
        prompts = {role: format_prompt(obs) for role, obs in observations.items()}
        outputs: dict[str, GenerationResult] = runtime.batch_decide(prompts)
        decisions: dict[str, AgentDecision] = {}
        for role, obs in observations.items():
            out = outputs[role]
            order = parse_order(
                out.text,
                fallback=obs.incoming_order_qty or DEFAULT_FALLBACK,
            )
            decisions[role] = AgentDecision(
                order_qty=order,
                tokens_used=out.tokens_used,
            )
        return decisions

    return _callback


def make_llm_baseline_team(
    runtime: LLMRuntime,
) -> tuple[dict[str, LLMBaselineAgent], PreTickCallback]:
    """Build the four-tier LLM baseline team + its batched callback."""
    agents: dict[str, LLMBaselineAgent] = {
        role: LLMBaselineAgent(runtime)
        for role in ("retailer", "wholesaler", "distributor", "factory")
    }
    return agents, make_llm_pre_tick_callback(runtime)


__all__ = [
    "DEFAULT_FALLBACK",
    "DEFAULT_MAX_ORDER",
    "LLMBaselineAgent",
    "SYSTEM_PROMPT",
    "format_prompt",
    "make_llm_baseline_team",
    "make_llm_pre_tick_callback",
    "parse_order",
]
