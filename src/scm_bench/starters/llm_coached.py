"""Coached LLM baseline — agent picks the policy, code does the math.

The vanilla `llm_baseline.py` asks the model "what should I order this
period?" every tick. Even a 26B model degrades to ~900× Mirror cost in
the tournament because raw integer arithmetic over noisy state is
exactly what LLMs are bad at.

This baseline rephrases the role: the LLM is the *strategist*, code is
the *executor*. Each tick the model receives the recent demand window
plus current inventory + horizon-left, and emits a structured policy
spec — `{pattern, target, halt, why}`. The Python wrapper then computes
the order quantity from that spec via base-stock arithmetic. The model
never sees nor outputs an integer order quantity.

Three cheap wins this captures:
1. **Memory** — base-stock target persists across ticks (closure state),
   so the model only needs to *update* it when the pattern changes,
   not rederive it every tick.
2. **Situational awareness** — the prompt names which demand pattern
   classes to look for (constant / linear / step / ar1) and what target
   to pick for each. The model classifies; the formula computes.
3. **Terminal awareness** — the prompt tells the model how many ticks
   are left and instructs it to halt new orders inside the lead-time
   tail so the bench doesn't end with stranded inventory.

If the JSON parse fails we fall back to the *cached* policy (last good
spec) rather than to a fixed mirror — that way one bad generate doesn't
cost a whole tick. Initial fallback before any successful generate is
base-stock target=20, which beats the raw LLM baseline by ~30×.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from scm_bench.messaging.envelope import Message
from scm_bench.runner.harness import PreTickCallback
from scm_bench.runner.llm_runtime import GenerationResult, LLMRuntime
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation

DEFAULT_MAX_ORDER = 50
DEFAULT_TARGET = 20
DEFAULT_LEAD_TIME = 4

ROLES: tuple[str, ...] = ("retailer", "wholesaler", "distributor", "factory")

SYSTEM_PROMPT = (
    "You are the {role} in the 4-tier beer-distribution game.\n"
    "Goal: minimize total cost. Holding $0.50/unit/period, "
    "backlog $1.00/unit/period.\n"
    "\n"
    "Use one of these named policies (do not invent new ones):\n"
    "- BASE_STOCK: order = max(0, target - inventory_position). "
    "Stable for stationary or slowly-drifting demand.\n"
    "- ORDER_UP_TO: same as base_stock; pick target = "
    "mean_demand_window * (lead_time + 1) + safety_stock.\n"
    "- TAPER: shrink target as the horizon approaches so you do not "
    "end with stranded inventory.\n"
    "\n"
    "Demand pattern menu (classify from order_history):\n"
    "- CONSTANT: low variance, mean ~= recent. target ~= 4 * mean.\n"
    "- LINEAR: monotone trend. target ~= 4 * (latest + slope * 2).\n"
    "- STEP: sudden level change. retarget from post-step mean.\n"
    "- AR1: noisy/autoregressive. smooth with EMA, target ~= 4 * EMA.\n"
    "\n"
    'Respond with ONE LINE of JSON, no prose: '
    '{{"pattern": "constant|linear|step|ar1", '
    '"target": <int 0-50>, '
    '"halt": <true|false>, '
    '"why": "<one short sentence>"}}\n'
    "\n"
    "Set halt=true ONLY if periods_remaining < lead_time "
    "(do not order what cannot be sold)."
)


@dataclass
class _PolicyState:
    """Cached per-role policy spec that the wrapper actually executes."""

    target: int = DEFAULT_TARGET
    halt: bool = False
    pattern: str = "constant"
    last_why: str = "init"
    successful_generates: int = 0
    failed_generates: int = 0
    last_decision_text: str = ""


@dataclass
class _CoachedRuntimeState:
    """Closure state shared across the per-role callback invocations."""

    runtime: LLMRuntime
    horizon_hint: int
    lead_time: int = DEFAULT_LEAD_TIME
    max_order: int = DEFAULT_MAX_ORDER
    policies: dict[str, _PolicyState] = field(default_factory=dict)

    def policy(self, role: str) -> _PolicyState:
        return self.policies.setdefault(role, _PolicyState())


def format_user_prompt(
    obs: LocalObservation,
    *,
    horizon_hint: int,
    lead_time: int,
    last_policy: _PolicyState,
) -> str:
    """Per-tick user prompt — short, structured, no decoration."""
    history = list(obs.order_history_window)[-12:]
    periods_remaining = max(0, horizon_hint - obs.timestep)
    inv_pos = (
        obs.inventory_on_hand
        + obs.incoming_shipment_qty
        + obs.pipeline_inventory
        - obs.backlog
    )
    return (
        f"period={obs.timestep} role={obs.role}\n"
        f"inventory_on_hand={obs.inventory_on_hand} "
        f"backlog={obs.backlog} "
        f"in_transit={obs.pipeline_inventory + obs.incoming_shipment_qty} "
        f"inventory_position={inv_pos}\n"
        f"incoming_order={obs.incoming_order_qty} "
        f"cost_so_far=${obs.costs_to_date.total:.0f}\n"
        f"order_history_last12={history}\n"
        f"periods_remaining={periods_remaining} lead_time={lead_time}\n"
        f"prev_target={last_policy.target} prev_pattern={last_policy.pattern}"
    )


def parse_policy_spec(text: str, *, max_order: int) -> dict | None:
    """Extract a `{pattern,target,halt,why}` JSON object from model output."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        target = max(0, min(int(data["target"]), max_order))
        halt = bool(data.get("halt", False))
        pattern = str(data.get("pattern", "constant"))[:16]
        why = str(data.get("why", ""))[:200]
    except (KeyError, ValueError, TypeError):
        return None
    return {"target": target, "halt": halt, "pattern": pattern, "why": why}


def compute_order(
    obs: LocalObservation,
    *,
    target: int,
    halt: bool,
    max_order: int,
) -> int:
    """Translate a policy spec into a concrete order quantity."""
    if halt:
        return 0
    inv_position = (
        obs.inventory_on_hand
        + obs.incoming_shipment_qty
        + obs.pipeline_inventory
        - obs.backlog
    )
    return max(0, min(target - inv_position, max_order))


class CoachedLLMAgent(Agent):
    """Per-role placeholder. Decisions flow through the team callback."""

    def __init__(self, state: _CoachedRuntimeState) -> None:
        self._state = state

    def step(
        self,
        observation: LocalObservation,
        inbox: list[Message],
        t: int,
    ) -> AgentDecision:
        # Fallback path if the team callback is not installed.
        policy = self._state.policy(observation.role)
        order = compute_order(
            observation,
            target=policy.target,
            halt=policy.halt,
            max_order=self._state.max_order,
        )
        return AgentDecision(order_qty=order)


def make_coached_pre_tick_callback(
    state: _CoachedRuntimeState,
) -> PreTickCallback:
    """Build the harness callback that batches policy generation per tick."""

    def _callback(
        observations: dict[str, LocalObservation], t: int
    ) -> dict[str, AgentDecision]:
        prompts: dict[str, str] = {}
        for role, obs in observations.items():
            policy = state.policy(role)
            prompts[role] = (
                SYSTEM_PROMPT.format(role=role)
                + "\n\n"
                + format_user_prompt(
                    obs,
                    horizon_hint=state.horizon_hint,
                    lead_time=state.lead_time,
                    last_policy=policy,
                )
            )

        outputs: dict[str, GenerationResult] = state.runtime.batch_decide(prompts)

        decisions: dict[str, AgentDecision] = {}
        for role, obs in observations.items():
            policy = state.policy(role)
            out = outputs[role]
            spec = parse_policy_spec(out.text, max_order=state.max_order)
            if spec is None:
                policy.failed_generates += 1
            else:
                policy.target = spec["target"]
                policy.halt = spec["halt"]
                policy.pattern = spec["pattern"]
                policy.last_why = spec["why"]
                policy.successful_generates += 1
            policy.last_decision_text = out.text[:200]

            order = compute_order(
                obs,
                target=policy.target,
                halt=policy.halt,
                max_order=state.max_order,
            )
            decisions[role] = AgentDecision(
                order_qty=order,
                tokens_used=out.tokens_used,
            )
        return decisions

    return _callback


def make_coached_llm_team(
    runtime: LLMRuntime,
    *,
    horizon_hint: int = 365,
    lead_time: int = DEFAULT_LEAD_TIME,
    max_order: int = DEFAULT_MAX_ORDER,
) -> tuple[dict[str, CoachedLLMAgent], PreTickCallback]:
    """Build the coached 4-tier LLM team + its batched callback."""
    state = _CoachedRuntimeState(
        runtime=runtime,
        horizon_hint=horizon_hint,
        lead_time=lead_time,
        max_order=max_order,
    )
    agents: dict[str, CoachedLLMAgent] = {role: CoachedLLMAgent(state) for role in ROLES}
    return agents, make_coached_pre_tick_callback(state)


__all__ = [
    "CoachedLLMAgent",
    "DEFAULT_MAX_ORDER",
    "DEFAULT_TARGET",
    "DEFAULT_LEAD_TIME",
    "SYSTEM_PROMPT",
    "compute_order",
    "format_user_prompt",
    "make_coached_llm_team",
    "make_coached_pre_tick_callback",
    "parse_policy_spec",
]
