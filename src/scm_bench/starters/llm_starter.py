"""Helpers for agent LLM-driven agent bundles.

The LLM starter ships four `agent.py` files that call this
module to (a) lazily build a single shared `LLMRuntime` per process
(default backend: Ollama, model `gemma4:e4b`), and (b) turn one
`(role, system_prompt, observation)` tuple into an `AgentDecision`
with a mirror-policy fallback so the bundle always validates.

Agents edit two things in `agent.py`:
- `SYSTEM_PROMPT` — the per-role brief sent to the model
- the body of `step()` — if they want to do anything beyond
  "ask the model for an order quantity"

Everything else (env-driven backend selection, batch-of-one fan-out,
order parsing, fallback) lives here so a five-line `step()` is enough
to get a working LLM agent.

Env vars (read once at first call):
- `SCB_LLM_BACKEND`   — default `ollama`
- `SCB_LLM_MODEL`     — default `gemma4:e4b`
- `SCB_LLM_BASE_URL`  — default `http://localhost:11434/v1`
- `SCB_LLM_API_KEY`   — optional bearer token
- `SCB_LLM_TIMEOUT_S` — default 60
"""

from __future__ import annotations

import os
from threading import Lock

from scm_bench.runner.llm_runtime import LLMRuntime
from scm_bench.sdk.contract import AgentDecision, LocalObservation
from scm_bench.starters.llm_baseline import (
    DEFAULT_FALLBACK,
    DEFAULT_MAX_ORDER,
    parse_order,
)

DEFAULT_STUDENT_MODEL = "gemma4:e4b"
DEFAULT_STUDENT_BACKEND = "ollama"

USER_TEMPLATE = (
    "timestep: {timestep}\n"
    "inventory_on_hand: {inventory_on_hand}\n"
    "backlog: {backlog}\n"
    "incoming_order_qty: {incoming_order_qty}\n"
    "incoming_shipment_qty: {incoming_shipment_qty}\n"
    "pipeline_inventory: {pipeline_inventory}\n"
    "costs_to_date: ${costs_to_date:.2f}\n"
    "\n"
    'Reply with ONLY a JSON object: {{"order": <int 0-{max_order}>, '
    '"why": "<one short sentence>"}}'
)

_RUNTIME: LLMRuntime | None = None
_RUNTIME_LOCK = Lock()


def make_runtime() -> LLMRuntime:
    """Lazy, process-wide `LLMRuntime` singleton."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = LLMRuntime(
                backend_name=os.environ.get("SCB_LLM_BACKEND", DEFAULT_STUDENT_BACKEND),
                model_id=os.environ.get("SCB_LLM_MODEL", DEFAULT_STUDENT_MODEL),
            )
    return _RUNTIME


def reset_runtime_for_tests() -> None:
    """Drop the cached runtime so tests can rebuild with a different backend."""
    global _RUNTIME
    _RUNTIME = None


def format_user_prompt(observation: LocalObservation, *, max_order: int = DEFAULT_MAX_ORDER) -> str:
    """Render the per-tick observation using the SDK field names verbatim.

    The labels intentionally match `LocalObservation`'s snake_case fields
    one-for-one so the SYSTEM_PROMPT roster and what the model actually
    sees are the same vocabulary. `incoming_shipment_qty` and
    `pipeline_inventory` are shown separately (not merged) for the same
    reason.
    """
    return USER_TEMPLATE.format(
        timestep=observation.timestep,
        inventory_on_hand=observation.inventory_on_hand,
        backlog=observation.backlog,
        incoming_order_qty=observation.incoming_order_qty,
        incoming_shipment_qty=observation.incoming_shipment_qty,
        pipeline_inventory=observation.pipeline_inventory,
        costs_to_date=observation.costs_to_date.total,
        max_order=max_order,
    )


def decide_with_llm(
    *,
    role: str,
    system_prompt: str,
    observation: LocalObservation,
    runtime: LLMRuntime | None = None,
    max_order: int = DEFAULT_MAX_ORDER,
) -> AgentDecision:
    """Single-prompt LLM decision with mirror-policy fallback.

    Builds one prompt, calls the runtime through its `batch_decide`
    surface (so the call path matches the aggregate baseline), parses an
    integer order from the response, and returns an `AgentDecision`.

    On any backend error (Ollama not running, network down, malformed
    response) we fall back to `incoming_order_qty` so the bundle still
    runs end-to-end and the validator's smoke test passes without an
    LLM in the loop.
    """
    runtime = runtime if runtime is not None else make_runtime()
    user = format_user_prompt(observation, max_order=max_order)
    prompt = f"{system_prompt.strip()}\n\n{user}"
    fallback = observation.incoming_order_qty or DEFAULT_FALLBACK
    try:
        out = runtime.batch_decide({role: prompt})[role]
    except Exception:  # noqa: BLE001 — bundle must keep running
        return AgentDecision(order_qty=max(0, min(fallback, max_order)), tokens_used=0)

    order = parse_order(out.text, max_order=max_order, fallback=fallback)
    return AgentDecision(order_qty=order, tokens_used=out.tokens_used)


__all__ = [
    "DEFAULT_STUDENT_BACKEND",
    "DEFAULT_STUDENT_MODEL",
    "USER_TEMPLATE",
    "decide_with_llm",
    "format_user_prompt",
    "make_runtime",
    "reset_runtime_for_tests",
]
