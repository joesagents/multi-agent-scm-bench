# Team Bundle — LLM starter (scm_bench)

Same four-agent contract as the Mirror starter, but each `step()`
queries a local LLM (default: Ollama running `gemma4:e4b`) for the
period's order quantity. Read the Mirror starter's `README.md` for the
contract, the forbiddens, and the workflow — that all still applies.

This README only covers what's different.

## What runs the LLM

The four `agent.py` files share one helper module from the wheel:

```python
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime
```

`make_runtime()` returns a process-wide singleton `LLMRuntime`. By
default it builds an `OpenAICompatBackend` pointing at
`http://localhost:11434/v1` (Ollama's OpenAI-compatible endpoint) with
model `gemma4:e4b`. Override with environment variables:

| Variable | Default | What it does |
|---|---|---|
| `SCB_LLM_BACKEND` | `ollama` | `ollama`, `openai_compat`, `transformers`, `vllm` |
| `SCB_LLM_MODEL` | `gemma4:e4b` | Any model the backend can serve |
| `SCB_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint root |
| `SCB_LLM_API_KEY` | *(unset)* | Bearer token, if needed |
| `SCB_LLM_TIMEOUT_S` | `60` | Per-request timeout |

## Setup (Ollama path)

```bash
# 1. Install Ollama from https://ollama.com (one-time)
# 2. Pull the model (once; ~9 GB for e4b, ~7 GB for e2b)
ollama pull gemma4:e4b

# 3. Start the server (Ollama usually does this automatically on
#    macOS; on Linux you may need `ollama serve &`)

# 4. Run your bundle as usual
scm_bench test-bundle .
scm_bench run-scenario --bundle . --scenario s1.1 \
    --out runs --batch-id me --team-id my-team
```

## Fallback behaviour (this matters)

If the LLM call fails — Ollama not running, model not pulled, network
hiccup — `decide_with_llm()` returns `AgentDecision(order_qty=
incoming_order_qty)`. That's the Mirror policy. So:

- The bundle **always validates** even without an LLM in the loop.
- A run with no LLM produces Mirror-baseline numbers (composite ≈
  1.000), not crashes.
- To verify the LLM actually fired, check `tokens_used` in the run
  metrics — it's `0` for fallback, positive for real calls.

## What to edit

Two places per agent file:

1. **`SYSTEM_PROMPT`** at the top — the brief sent to the model. The
   default is short and role-specific; sharpen it to match the policy
   you want.
2. **`step()`** — if you want anything beyond "ask the model what to
   order this period." Examples: pre-process the observation,
   post-process the order with a sanity clamp, mix LLM and base-stock
   based on horizon-left, cache responses across ticks of stable
   demand.

## Files

```
team_bundle/
├── manifest.json
├── retailer/{agent.py, agent.yaml}      # SYSTEM_PROMPT for retailer
├── wholesaler/{agent.py, agent.yaml}    # SYSTEM_PROMPT for wholesaler
├── distributor/{agent.py, agent.yaml}   # SYSTEM_PROMPT for distributor
├── factory/{agent.py, agent.yaml}       # SYSTEM_PROMPT for factory
├── tests/test_local.py
└── AGENTS.md                            # contract for in-IDE coding agents
```
