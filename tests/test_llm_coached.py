"""Coached LLM baseline tests.

The "coached" baseline asks the LLM for a *policy spec* and lets Python
do the integer arithmetic. We verify the deterministic plumbing here
(parse, compute, callback wiring) using a `MockBackend`; the real model
is exercised on the cluster.

Three slices:

1. `parse_policy_spec` accepts well-formed JSON, clamps the target,
   and returns ``None`` for unparseable text (so the wrapper falls
   back to the cached spec).
2. `compute_order` honours `halt`, clamps to `[0, max_order]`, and
   uses inventory_position = on_hand + in_transit + pipeline − backlog.
3. The pre-tick callback batches all four roles into one backend call
   and propagates `tokens_used` into `AgentDecision`. A failed parse
   degrades to the cached `_PolicyState` rather than crashing the run.
"""

from __future__ import annotations

import pytest

from scm_bench.runner.llm_runtime import (
    GenerationResult,
    LLMBackend,
    LLMRuntime,
)
from scm_bench.sdk.contract import LocalObservation
from scm_bench.starters.llm_coached import (
    DEFAULT_MAX_ORDER,
    DEFAULT_TARGET,
    compute_order,
    make_coached_llm_team,
    parse_policy_spec,
)


class MockBackend(LLMBackend):
    """Returns a canned output per call; records every call's prompts."""

    def __init__(self, *, text: str, tokens: int = 11) -> None:
        self._text = text
        self._tokens = tokens
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "mock-coached"

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        self.calls.append(list(prompts))
        return [GenerationResult(text=self._text, tokens_used=self._tokens) for _ in prompts]


def _obs(role: str, *, on_hand: int, backlog: int = 0,
         incoming_order: int = 4, incoming_shipment: int = 4,
         pipeline: int = 4, t: int = 0) -> LocalObservation:
    return LocalObservation(
        timestep=t,
        role=role,
        inventory_on_hand=on_hand,
        backlog=backlog,
        incoming_order_qty=incoming_order,
        incoming_shipment_qty=incoming_shipment,
        pipeline_inventory=pipeline,
    )


def test_parse_policy_spec_accepts_valid_json() -> None:
    spec = parse_policy_spec(
        '{"pattern": "constant", "target": 22, "halt": false, "why": "stable"}',
        max_order=DEFAULT_MAX_ORDER,
    )
    assert spec == {"pattern": "constant", "target": 22, "halt": False, "why": "stable"}


def test_parse_policy_spec_clamps_target_to_max_order() -> None:
    spec = parse_policy_spec(
        '{"pattern": "linear", "target": 999, "halt": false, "why": "spike"}',
        max_order=DEFAULT_MAX_ORDER,
    )
    assert spec is not None and spec["target"] == DEFAULT_MAX_ORDER


def test_parse_policy_spec_returns_none_on_garbage() -> None:
    assert parse_policy_spec("no JSON here", max_order=DEFAULT_MAX_ORDER) is None
    assert parse_policy_spec('{"target": "not-an-int"}', max_order=DEFAULT_MAX_ORDER) is None


def test_parse_policy_spec_extracts_object_from_prose() -> None:
    text = 'Sure, here is the spec: {"pattern":"step","target":18,"halt":false,"why":"ok"} done.'
    spec = parse_policy_spec(text, max_order=DEFAULT_MAX_ORDER)
    assert spec is not None and spec["target"] == 18 and spec["pattern"] == "step"


def test_compute_order_honours_halt() -> None:
    obs = _obs("retailer", on_hand=2, backlog=0, pipeline=0, incoming_shipment=0)
    assert compute_order(obs, target=20, halt=True, max_order=DEFAULT_MAX_ORDER) == 0


def test_compute_order_uses_inventory_position() -> None:
    # inv_position = 10 + 4 + 4 - 0 = 18; target 22 → order 4
    obs = _obs("retailer", on_hand=10, backlog=0, pipeline=4, incoming_shipment=4)
    assert compute_order(obs, target=22, halt=False, max_order=DEFAULT_MAX_ORDER) == 4


def test_compute_order_clamps_negative_to_zero() -> None:
    # inv_position 30; target 20 → -10 → 0
    obs = _obs("retailer", on_hand=20, pipeline=5, incoming_shipment=5)
    assert compute_order(obs, target=20, halt=False, max_order=DEFAULT_MAX_ORDER) == 0


def test_compute_order_clamps_to_max_order() -> None:
    obs = _obs("retailer", on_hand=0, pipeline=0, incoming_shipment=0, backlog=200)
    assert compute_order(obs, target=20, halt=False, max_order=DEFAULT_MAX_ORDER) == DEFAULT_MAX_ORDER


def test_callback_batches_one_call_per_tick() -> None:
    backend = MockBackend(
        text='{"pattern":"constant","target":20,"halt":false,"why":"ok"}',
        tokens=11,
    )
    runtime = LLMRuntime(model_id="mock-coached", backend=backend)
    _agents, callback = make_coached_llm_team(runtime, horizon_hint=30)

    obs = {role: _obs(role, on_hand=12) for role in
           ("retailer", "wholesaler", "distributor", "factory")}
    decisions = callback(obs, 0)

    assert set(decisions.keys()) == set(obs.keys())
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 4
    for d in decisions.values():
        assert d.tokens_used == 11
    # inv_position 12+4+4-0 = 20; target 20 → order 0
    for d in decisions.values():
        assert d.order_qty == 0


def test_callback_falls_back_to_cached_policy_on_bad_parse() -> None:
    backend = MockBackend(text="not json at all", tokens=2)
    runtime = LLMRuntime(model_id="mock-coached", backend=backend)
    _agents, callback = make_coached_llm_team(runtime, horizon_hint=30)

    obs = {role: _obs(role, on_hand=8) for role in
           ("retailer", "wholesaler", "distributor", "factory")}
    decisions = callback(obs, 0)

    # Default cached target=20; inv_position 8+4+4-0=16; order = 4
    for role, d in decisions.items():
        assert d.order_qty == DEFAULT_TARGET - 16, role
        assert d.tokens_used == 2


def test_callback_persists_target_across_ticks() -> None:
    backend = MockBackend(
        text='{"pattern":"linear","target":35,"halt":false,"why":"trend"}',
        tokens=9,
    )
    runtime = LLMRuntime(model_id="mock-coached", backend=backend)
    _agents, callback = make_coached_llm_team(runtime, horizon_hint=30)

    obs1 = {r: _obs(r, on_hand=10, t=0) for r in
            ("retailer", "wholesaler", "distributor", "factory")}
    callback(obs1, 0)

    # Second tick: backend now returns garbage. Cached target=35 must be reused.
    backend._text = "garbage"
    obs2 = {r: _obs(r, on_hand=10, t=1) for r in
            ("retailer", "wholesaler", "distributor", "factory")}
    decisions = callback(obs2, 1)

    # inv_position 10+4+4-0=18; cached target=35 → order 17
    for d in decisions.values():
        assert d.order_qty == 17


def test_coached_team_via_batch_run(tmp_path) -> None:
    """End-to-end: coached LLM team produced via batch.coached_llm_baseline_team_spec
    runs through batch_run, emits a row, and one batched call per tick."""
    from scm_bench.runner.batch import (
        TeamSpec,
        batch_run,
        mirror_team_spec,
    )
    from scm_bench.trace.store import RunStore

    backend = MockBackend(
        text='{"pattern":"constant","target":18,"halt":false,"why":"steady"}',
        tokens=10,
    )

    def builder():
        runtime = LLMRuntime(model_id="mock-coached", backend=backend)
        agents, cb = make_coached_llm_team(runtime, horizon_hint=30)
        return dict(agents), cb

    builder.__name__ = "mock_coached_builder"

    store = RunStore(tmp_path / "runs.db")
    summary = batch_run(
        batch_id="t-coached",
        teams=[
            mirror_team_spec(),
            TeamSpec(team_id="__coached_mock__", team_builder=builder),
        ],
        scenario_ids=["intro_step_demand"],
        seeds=[0],
        store=store,
        include_mirror=False,
    )

    coached = [c for c in summary.cells if c.team_id == "__coached_mock__"]
    assert len(coached) == 1
    cell = coached[0]
    assert cell.ok, cell.error
    # Only the coached cell hits the mock backend; mirror uses MirrorAgent.
    # 30 ticks × 1 batched call per tick.
    assert len(backend.calls) == 30
    for prompts in backend.calls:
        assert len(prompts) == 4
    assert cell.row is not None
    assert cell.row.tokens_used == 4 * 30 * 10
