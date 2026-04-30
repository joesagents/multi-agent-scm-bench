"""Per-tick orchestrator.

Connects four agents to one Environment for one scenario × one seed.

Responsibilities:
- Build LocalObservation from engine state at the start of each tick.
- Optionally call `pre_tick_callback(observations, t)` — a batch hook
  used by teams that need to compute decisions for several roles in a
  single fan-out (the LLM baseline shares one `model.generate` across
  all four tiers; without the hook, sequential `agent.step()` calls
  would be ~4× slower).
- Call agent.step(obs, inbox, t) for any role the callback did not
  precompute.
- Validate the returned AgentDecision (Pydantic does this on construction).
- If `agent_manifests` is supplied, enforce the manifest contract per
  tick (see runner/contract.py): emitted messages must match
  declared supports_messages, and exposed memory must stay within
  declared memory_max_entries. A violation aborts the run with
  E_CONTRACT_VIOLATION.
- Pull order_qty into env.step().
- Collect (but do not deliver) any messages an agent returns.

Still deferred:
- typed message bus delivery + delay/drop/budget enforcement
- supports_tools enforcement (no tool-call surface in AgentDecision yet)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scm_bench.engine.environment import (
    Environment,
    SimulationResult,
)
from scm_bench.engine.state import TIER_NAMES
from scm_bench.messaging.envelope import Message
from scm_bench.runner.contract import check_memory, check_messages
from scm_bench.scenarios.scenario import Scenario
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import (
    AgentDecision,
    CostBreakdown,
    LocalObservation,
)
from scm_bench.sdk.manifest import AgentManifest


PreTickCallback = Callable[
    [dict[str, "LocalObservation"], int],
    dict[str, "AgentDecision"],
]
"""Optional batch hook: takes the per-role observations for tick `t` and
returns a dict of role → AgentDecision for any roles whose decision was
computed in the batch. Roles missing from the returned dict fall back to
their own `agent.step()`. Used by the LLM baseline to share one
`model.generate` across all four tiers."""


@dataclass
class TickRecord:
    t: int
    consumer_demand: int
    observations: dict[str, LocalObservation]
    decisions: dict[str, AgentDecision]
    messages_emitted: list[Message]


@dataclass
class RunRecord:
    run_id: str
    scenario_id: str
    seed: int
    horizon: int
    ticks: list[TickRecord] = field(default_factory=list)
    result: SimulationResult | None = None


def _build_observation(
    *,
    role: str,
    env: Environment,
    t: int,
    consumer_demand: int,
    window: int,
) -> LocalObservation:
    tier = env.state.get_tier(role)

    if role == "retailer":
        incoming_order_qty = consumer_demand
    else:
        incoming_order_qty = tier.incoming_orders[0]

    incoming_shipment_qty = tier.incoming_shipments[0]
    pipeline_inventory = sum(tier.incoming_shipments[1:])

    return LocalObservation(
        timestep=t,
        role=role,
        inventory_on_hand=tier.inventory,
        backlog=tier.backlog,
        incoming_order_qty=incoming_order_qty,
        incoming_shipment_qty=incoming_shipment_qty,
        pipeline_inventory=pipeline_inventory,
        order_history_window=list(tier.order_history[-window:]),
        shipment_history_window=list(tier.shipment_history[-window:]),
        costs_to_date=CostBreakdown(
            holding=tier.cumulative_holding_cost,
            stockout=tier.cumulative_backlog_cost,
            total=tier.cumulative_cost,
        ),
    )


def run_one(
    *,
    agents: dict[str, Agent],
    scenario: Scenario,
    seed: int,
    run_id: str,
    max_ticks: int | None = None,
    agent_configs: dict[str, dict[str, Any]] | None = None,
    agent_manifests: dict[str, AgentManifest] | None = None,
    pre_tick_callback: PreTickCallback | None = None,
) -> RunRecord:
    """Run one bundle on one scenario × one seed.

    Args:
        agents: role -> Agent instance (already validated)
        scenario: scenario to run
        seed: seed for the demand function and any agent randomness
        run_id: identifier for the resulting RunRecord
        max_ticks: optional early-stop (used by validator's smoke run)
        agent_configs: per-role config dict passed to agent.reset()
        agent_manifests: optional per-role AgentManifest. If supplied,
            the harness enforces the manifest contract every tick
            (supports_messages, memory_max_entries). Pass `None` only
            for built-in starters / mirror baselines whose contract is
            trusted.
        pre_tick_callback: optional batch hook called once per tick
            with all four observations. Roles for which it returns an
            AgentDecision skip the per-role `agent.step()` call. Used
            by the LLM baseline to share one batched `model.generate`
            across all four tiers.
    """
    missing = set(TIER_NAMES) - set(agents)
    if missing:
        raise ValueError(f"agents dict missing roles: {sorted(missing)}")

    cfgs = dict(agent_configs or {})
    for role, agent in agents.items():
        agent.reset(role=role, config=cfgs.get(role, {}), seed=seed)

    env = Environment(scenario.env_config)
    env.reset()

    horizon = scenario.horizon if max_ticks is None else min(scenario.horizon, max_ticks)
    record = RunRecord(
        run_id=run_id,
        scenario_id=scenario.id,
        seed=seed,
        horizon=horizon,
    )

    inboxes: dict[str, list[Message]] = {role: [] for role in TIER_NAMES}

    for t in range(horizon):
        consumer_demand = int(scenario.demand_fn(t, seed))

        observations: dict[str, LocalObservation] = {}
        for role in TIER_NAMES:
            observations[role] = _build_observation(
                role=role,
                env=env,
                t=t,
                consumer_demand=consumer_demand,
                window=scenario.observation_window,
            )

        precomputed: dict[str, AgentDecision] = {}
        if pre_tick_callback is not None:
            precomputed = dict(pre_tick_callback(observations, t) or {})

        decisions: dict[str, AgentDecision] = {}
        emitted: list[Message] = []
        for role in TIER_NAMES:
            if role in precomputed:
                decision = precomputed[role]
            else:
                decision = agents[role].step(observations[role], list(inboxes[role]), t)
            if not isinstance(decision, AgentDecision):
                raise TypeError(
                    f"agent {role!r} returned {type(decision).__name__}, "
                    "expected AgentDecision"
                )
            decisions[role] = decision
            if agent_manifests is not None and role in agent_manifests:
                manifest = agent_manifests[role]
                check_messages(
                    role=role, manifest=manifest, messages=decision.messages, t=t
                )
                check_memory(
                    role=role, manifest=manifest, agent=agents[role], t=t
                )
            for msg in decision.messages:
                msg.sender_role = role
                msg.ts_sent = t
                emitted.append(msg)

        record.ticks.append(
            TickRecord(
                t=t,
                consumer_demand=consumer_demand,
                observations=observations,
                decisions=decisions,
                messages_emitted=emitted,
            )
        )

        order_qtys = {role: int(decisions[role].order_qty) for role in TIER_NAMES}
        done, _info = env.step(consumer_demand, order_qtys)
        if done:
            break

    record.result = env.get_result()
    return record
