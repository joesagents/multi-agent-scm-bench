"""Tests for the agent-bundle LLM helpers.

Two surfaces to cover:

1. `decide_with_llm` — happy path (LLM returns parseable JSON), parse
   failure, and backend exception (must fall back to mirror).
2. The shipped LLM starter bundle template — must validate end-to-end
   via `validate_bundle` without a running Ollama (the fallback path
   keeps the smoke test green).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scm_bench.starters.llm_starter as llm_starter
from scm_bench.runner.llm_runtime import (
    GenerationResult,
    LLMBackend,
    LLMRuntime,
)
from scm_bench.sdk.contract import LocalObservation
from scm_bench.sdk.validator import ValidationError, validate_bundle
from scm_bench.starters.llm_starter import decide_with_llm, format_user_prompt


class _CannedBackend(LLMBackend):
    def __init__(self, text: str = '{"order": 6}', tokens: int = 9) -> None:
        self._text = text
        self._tokens = tokens
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "canned"

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        self.calls.append(list(prompts))
        return [GenerationResult(text=self._text, tokens_used=self._tokens) for _ in prompts]


class _ExplodingBackend(LLMBackend):
    @property
    def model_id(self) -> str:
        return "boom"

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        raise RuntimeError("backend down")


def _obs(role: str = "retailer", *, incoming: int = 11) -> LocalObservation:
    return LocalObservation(
        timestep=3,
        role=role,
        inventory_on_hand=10,
        backlog=2,
        incoming_order_qty=incoming,
        incoming_shipment_qty=4,
        pipeline_inventory=4,
    )


@pytest.fixture(autouse=True)
def _reset_runtime_singleton():
    llm_starter.reset_runtime_for_tests()
    yield
    llm_starter.reset_runtime_for_tests()


def test_decide_with_llm_uses_parsed_order_and_propagates_tokens() -> None:
    backend = _CannedBackend(text='{"order": 6, "why": "smooth"}', tokens=14)
    runtime = LLMRuntime(model_id="canned", backend=backend)
    decision = decide_with_llm(
        role="retailer",
        system_prompt="You are the retailer.",
        observation=_obs(),
        runtime=runtime,
    )
    assert decision.order_qty == 6
    assert decision.tokens_used == 14
    assert len(backend.calls) == 1
    assert "You are the retailer." in backend.calls[0][0]


def test_decide_with_llm_clamps_to_max_order() -> None:
    backend = _CannedBackend(text='{"order": 999}')
    runtime = LLMRuntime(model_id="canned", backend=backend)
    decision = decide_with_llm(
        role="factory",
        system_prompt="x",
        observation=_obs("factory"),
        runtime=runtime,
        max_order=50,
    )
    assert decision.order_qty == 50


def test_decide_with_llm_falls_back_to_mirror_on_unparseable_text() -> None:
    backend = _CannedBackend(text="model said: be cautious", tokens=4)
    runtime = LLMRuntime(model_id="canned", backend=backend)
    decision = decide_with_llm(
        role="retailer",
        system_prompt="x",
        observation=_obs(incoming=11),
        runtime=runtime,
    )
    assert decision.order_qty == 11  # incoming_order_qty
    assert decision.tokens_used == 4


def test_decide_with_llm_falls_back_to_mirror_on_backend_error() -> None:
    runtime = LLMRuntime(model_id="canned", backend=_ExplodingBackend())
    decision = decide_with_llm(
        role="retailer",
        system_prompt="x",
        observation=_obs(incoming=7),
        runtime=runtime,
    )
    assert decision.order_qty == 7
    assert decision.tokens_used == 0


def test_make_runtime_is_a_singleton(monkeypatch) -> None:
    """Two consecutive calls return the same object; only one backend init."""
    captured: list[dict] = []

    def fake_init(self, *, model_id, backend=None, backend_name=None, max_new_tokens=48):
        captured.append({"model_id": model_id, "backend_name": backend_name})
        # bypass real backend construction
        self._model_id = model_id
        self._backend = _CannedBackend()

    monkeypatch.setattr(LLMRuntime, "__init__", fake_init)
    monkeypatch.setenv("SCB_LLM_BACKEND", "ollama")
    monkeypatch.setenv("SCB_LLM_MODEL", "gemma4:e2b")

    a = llm_starter.make_runtime()
    b = llm_starter.make_runtime()
    assert a is b
    assert len(captured) == 1
    assert captured[0] == {"model_id": "gemma4:e2b", "backend_name": "ollama"}


def test_user_prompt_uses_sdk_field_names_and_keeps_fields_separate() -> None:
    """The user prompt must use the same labels the SYSTEM_PROMPTs list.

    Regression: before, the prompt merged `pipeline_inventory +
    incoming_shipment_qty` into a single `in_transit` line and used
    prose labels (`Inventory on hand:`) — divergent from the
    SYSTEM_PROMPT's snake_case roster. Don't let that come back.
    """
    obs = LocalObservation(
        timestep=7,
        role="wholesaler",
        inventory_on_hand=15,
        backlog=2,
        incoming_order_qty=9,
        incoming_shipment_qty=5,
        pipeline_inventory=4,
    )
    prompt = format_user_prompt(obs)

    # Each SDK field appears with its snake_case name and own value.
    assert "timestep: 7" in prompt
    assert "inventory_on_hand: 15" in prompt
    assert "backlog: 2" in prompt
    assert "incoming_order_qty: 9" in prompt
    assert "incoming_shipment_qty: 5" in prompt
    assert "pipeline_inventory: 4" in prompt
    assert "costs_to_date:" in prompt

    # The two transit-related fields stay split, not summed.
    assert "in_transit" not in prompt.lower()
    assert ": 9\n" in prompt or "incoming_shipment_qty: 5\npipeline_inventory: 4\n" in prompt


def test_llm_starter_template_validates_without_ollama(tmp_path) -> None:
    """The shipped LLM starter bundle template must validate end-to-end.

    No Ollama runs in CI / on dev laptops by default. The mirror
    fallback inside `decide_with_llm` keeps the smoke test green.
    """
    import shutil

    src = Path(
        "src/scm_bench/sdk/starter_template_llm"
    ).resolve()
    assert src.exists(), src
    bundle = tmp_path / "team_bundle"
    shutil.copytree(src, bundle)

    try:
        report = validate_bundle(bundle, smoke_ticks=5)
    except ValidationError as e:
        pytest.fail(f"LLM starter template did not validate [{e.code}]: {e}")
    assert report.smoke_run_ok, f"smoke not ok: {report.smoke_error}"
