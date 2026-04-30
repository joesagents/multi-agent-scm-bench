"""LLM baseline + per-tick batch hook tests.

Neither HF nor vLLM runs Gemma on macOS / CPU, so this file exercises
the runtime via a `MockBackend` that returns canned outputs. We verify:

1. The pre-tick callback batches all four roles into one backend call.
2. Token counts and order quantities are propagated into AgentDecision.
3. Bad / unparseable outputs degrade to the mirror fallback rather than
   crashing the run.
4. The harness wires the callback correctly: every tick gets a
   batched call, and `agent.step()` is bypassed for callback-served
   roles.
"""

from __future__ import annotations

from typing import Any

import pytest

from scm_bench.engine.environment import EnvironmentConfig
from scm_bench.runner.harness import run_one
from scm_bench.runner.llm_runtime import (
    GenerationResult,
    LLMBackend,
    LLMRuntime,
)
from scm_bench.scenarios.builtin import INTRO_STEP_DEMAND
from scm_bench.scenarios.scenario import Scenario
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.contract import AgentDecision, LocalObservation
from scm_bench.starters.llm_baseline import (
    LLMBaselineAgent,
    format_prompt,
    make_llm_baseline_team,
    make_llm_pre_tick_callback,
    parse_order,
)


class MockBackend(LLMBackend):
    """Returns a canned output per call; records every call's prompts."""

    def __init__(self, *, text: str = '{"order": 4, "reasoning": "mock"}', tokens: int = 12) -> None:
        self._text = text
        self._tokens = tokens
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "mock"

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        self.calls.append(list(prompts))
        return [GenerationResult(text=self._text, tokens_used=self._tokens) for _ in prompts]


class _TripwireAgent(Agent):
    """Asserts its `step()` is never called when the callback covers all roles."""

    def __init__(self) -> None:
        self.step_calls = 0

    def step(
        self, observation: LocalObservation, inbox: list[Any], t: int
    ) -> AgentDecision:
        self.step_calls += 1
        return AgentDecision(order_qty=0)


def _short_scenario() -> Scenario:
    """5-tick variant of INTRO_STEP_DEMAND for fast tests."""
    return Scenario(
        id="intro_step_demand_short",
        name="short",
        description="short smoke",
        demand_fn=INTRO_STEP_DEMAND.demand_fn,
        env_config=EnvironmentConfig(
            horizon=5,
            initial_inventory=12,
            initial_pipeline=(4, 4),
            initial_orders=(4, 4),
            holding_cost=0.50,
            backlog_cost=1.00,
        ),
        observation_window=8,
    )


def test_parse_order_extracts_json_payload() -> None:
    assert parse_order('{"order": 7, "reasoning": "ok"}') == 7


def test_parse_order_clamps_to_max() -> None:
    assert parse_order('{"order": 999}') == 50


def test_parse_order_falls_back_to_loose_integer() -> None:
    assert parse_order("model says order 6 units") == 6


def test_parse_order_fallback_when_no_signal() -> None:
    assert parse_order("no idea", fallback=8) == 8


def test_runtime_batch_decide_preserves_role_order() -> None:
    backend = MockBackend()
    runtime = LLMRuntime(model_id="mock", backend=backend)
    out = runtime.batch_decide(
        {"retailer": "p1", "wholesaler": "p2", "distributor": "p3", "factory": "p4"}
    )
    assert list(out.keys()) == ["retailer", "wholesaler", "distributor", "factory"]
    assert backend.calls == [["p1", "p2", "p3", "p4"]]


def test_pre_tick_callback_runs_one_batch_per_tick() -> None:
    backend = MockBackend()
    runtime = LLMRuntime(model_id="mock", backend=backend)
    agents, callback = make_llm_baseline_team(runtime)
    # Replace agents with tripwires so we catch any sequential `step()` calls.
    tripwires = {role: _TripwireAgent() for role in agents}

    record = run_one(
        agents=tripwires,
        scenario=_short_scenario(),
        seed=0,
        run_id="t-llm-batch",
        pre_tick_callback=callback,
    )

    assert len(record.ticks) == 5
    assert len(backend.calls) == 5
    for prompts in backend.calls:
        assert len(prompts) == 4
    for tw in tripwires.values():
        assert tw.step_calls == 0
    for tick in record.ticks:
        for role, decision in tick.decisions.items():
            assert decision.order_qty == 4
            assert decision.tokens_used == 12


def test_pre_tick_callback_falls_back_to_step_for_missing_roles() -> None:
    backend = MockBackend()
    runtime = LLMRuntime(model_id="mock", backend=backend)
    agents, _full_cb = make_llm_baseline_team(runtime)

    def partial_cb(observations, t):
        # Only cover two roles — the harness must call step() for the others.
        return {
            "retailer": AgentDecision(order_qty=3, tokens_used=5),
            "factory": AgentDecision(order_qty=9, tokens_used=7),
        }

    tripwires = {role: _TripwireAgent() for role in agents}
    record = run_one(
        agents=tripwires,
        scenario=_short_scenario(),
        seed=0,
        run_id="t-llm-partial",
        pre_tick_callback=partial_cb,
    )

    assert tripwires["wholesaler"].step_calls == 5
    assert tripwires["distributor"].step_calls == 5
    assert tripwires["retailer"].step_calls == 0
    assert tripwires["factory"].step_calls == 0
    for tick in record.ticks:
        assert tick.decisions["retailer"].order_qty == 3
        assert tick.decisions["factory"].order_qty == 9


def test_baseline_agent_fallback_path_uses_runtime() -> None:
    """If the harness has no callback wired, `agent.step()` must still work."""
    backend = MockBackend(text='{"order": 6}', tokens=20)
    runtime = LLMRuntime(model_id="mock", backend=backend)
    agent = LLMBaselineAgent(runtime)
    agent.reset(role="retailer", config={}, seed=0)

    obs = LocalObservation(
        timestep=0,
        role="retailer",
        inventory_on_hand=10,
        backlog=0,
        incoming_order_qty=8,
        incoming_shipment_qty=4,
        pipeline_inventory=4,
    )
    decision = agent.step(obs, [], 0)
    assert decision.order_qty == 6
    assert decision.tokens_used == 20


def test_unparseable_output_falls_back_to_incoming_order() -> None:
    backend = MockBackend(text="garbage with no number", tokens=4)
    runtime = LLMRuntime(model_id="mock", backend=backend)
    _agents, callback = make_llm_baseline_team(runtime)

    obs = {
        role: LocalObservation(
            timestep=0,
            role=role,
            inventory_on_hand=10,
            backlog=0,
            incoming_order_qty=11,
            incoming_shipment_qty=4,
            pipeline_inventory=0,
        )
        for role in ("retailer", "wholesaler", "distributor", "factory")
    }
    decisions = callback(obs, 0)
    for role, decision in decisions.items():
        assert decision.order_qty == 11, role
        assert decision.tokens_used == 4


def test_format_prompt_contains_role_and_observation_fields() -> None:
    obs = LocalObservation(
        timestep=3,
        role="wholesaler",
        inventory_on_hand=15,
        backlog=2,
        incoming_order_qty=9,
        incoming_shipment_qty=5,
        pipeline_inventory=4,
    )
    prompt = format_prompt(obs)
    assert "Tier:wholesaler" in prompt
    assert "OnHand:15" in prompt
    assert "Backlog:2" in prompt
    assert "OrdersIn:9" in prompt
    # in_transit aggregates incoming + pipeline (= 9 here)
    assert "InTransit:9" in prompt


def test_runtime_rejects_backend_size_mismatch() -> None:
    class BadBackend(LLMBackend):
        @property
        def model_id(self) -> str:
            return "bad"

        def generate(self, prompts):
            return [GenerationResult(text="x", tokens_used=1)]  # always 1, regardless of input

    runtime = LLMRuntime(model_id="bad", backend=BadBackend())
    with pytest.raises(RuntimeError, match="returned 1 outputs for 4 prompts"):
        runtime.batch_decide({r: "p" for r in ("retailer", "wholesaler", "distributor", "factory")})


def test_make_llm_pre_tick_callback_is_callable_independently() -> None:
    """The callback factory works without going through the full team builder."""
    backend = MockBackend()
    runtime = LLMRuntime(model_id="mock", backend=backend)
    cb = make_llm_pre_tick_callback(runtime)

    obs = {
        role: LocalObservation(
            timestep=0,
            role=role,
            inventory_on_hand=8,
            backlog=0,
            incoming_order_qty=4,
            incoming_shipment_qty=4,
            pipeline_inventory=4,
        )
        for role in ("retailer", "wholesaler", "distributor", "factory")
    }
    decisions = cb(obs, 0)
    assert set(decisions.keys()) == {"retailer", "wholesaler", "distributor", "factory"}
    assert all(d.order_qty == 4 for d in decisions.values())


def test_llm_team_runs_via_batch_run(tmp_path) -> None:
    """End-to-end: a team_builder TeamSpec ingests through batch_run +
    RunStore, the row carries non-zero tokens_used, and the harness
    runs exactly one batched call per tick (not four)."""
    from scm_bench.runner.batch import TeamSpec, batch_run, mirror_team_spec
    from scm_bench.starters.llm_baseline import make_llm_baseline_team
    from scm_bench.trace.store import RunStore

    backend = MockBackend(text='{"order": 5, "reasoning": "test"}', tokens=15)

    def builder():
        runtime = LLMRuntime(model_id="mock", backend=backend)
        agents, cb = make_llm_baseline_team(runtime)
        return dict(agents), cb

    builder.__name__ = "mock_llm_builder"

    store = RunStore(tmp_path / "runs.db")
    summary = batch_run(
        batch_id="t-llm-class",
        teams=[
            mirror_team_spec(),
            TeamSpec(team_id="__llm_mock__", team_builder=builder),
        ],
        scenario_ids=["intro_step_demand"],
        seeds=[0],
        store=store,
        include_mirror=False,  # mirror already in teams
    )

    llm_cells = [c for c in summary.cells if c.team_id == "__llm_mock__"]
    assert len(llm_cells) == 1
    cell = llm_cells[0]
    assert cell.ok, cell.error
    # 30 ticks × 1 batched call per tick = 30 calls (vs 30×4=120 if sequential).
    # The mirror cell also fires through MockBackend? No — mirror uses its own
    # MirrorAgent, which doesn't touch the runtime. So all 30 calls are LLM.
    assert len(backend.calls) == 30
    for prompts in backend.calls:
        assert len(prompts) == 4
    # tokens_used summed across 4 tiers × 30 ticks × 15 tokens = 1800
    assert cell.row is not None
    assert cell.row.tokens_used == 4 * 30 * 15
