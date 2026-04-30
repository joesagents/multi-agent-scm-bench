# AGENTS.md — scm_bench team bundle (LLM starter)

Contract for in-IDE coding agents (Claude Code, Codex, Cursor, etc.).
This bundle is the LLM-driven starter — every `step()` queries a local
LLM by default. The Mirror starter's `AGENTS.md` covers the contract
that's identical between bundles; read it for layout, hard constraints,
the SDK surface, the CLI commands. This file only documents the LLM
delta.

## Bundle shape (same as Mirror)

```
team_bundle/
├── manifest.json
├── retailer/{agent.py,agent.yaml}      # SYSTEM_PROMPT for retailer
├── wholesaler/{agent.py,agent.yaml}    # SYSTEM_PROMPT for wholesaler
├── distributor/{agent.py,agent.yaml}   # SYSTEM_PROMPT for distributor
├── factory/{agent.py,agent.yaml}       # SYSTEM_PROMPT for factory
├── tests/test_local.py
└── AGENTS.md                           # this file
```

## What every `step()` does

```python
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime

class RetailerAgent(Agent):
    def reset(self, *, role, config, seed):
        super().reset(role=role, config=config, seed=seed)
        self._runtime = make_runtime()      # process-wide singleton

    def step(self, observation, inbox, t):
        return decide_with_llm(
            role="retailer",
            system_prompt=SYSTEM_PROMPT,    # editable above
            observation=observation,
            runtime=self._runtime,
        )
```

`decide_with_llm` builds a single chat prompt
(`system_prompt + observation summary`), POSTs it to the configured
backend, parses the integer order from the JSON response, and returns
`AgentDecision(order_qty=int, tokens_used=int)`. On any backend error
it falls back to `AgentDecision(order_qty=incoming_order_qty)` — the
mirror policy.

## Backend resolution

`make_runtime()` reads env vars at first call (process-wide singleton):

| Variable | Default | Notes |
|---|---|---|
| `SCB_LLM_BACKEND` | `ollama` | also: `openai_compat`, `transformers`, `vllm` |
| `SCB_LLM_MODEL` | `gemma4:e4b` | any model the backend can serve |
| `SCB_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `SCB_LLM_API_KEY` | unset | bearer token for hosted endpoints |
| `SCB_LLM_TIMEOUT_S` | 60 | per-request timeout |

`ollama` is an alias for `openai_compat` with the Ollama default URL.

## Setup the user must do once

```bash
# 1. Install Ollama: https://ollama.com (Mac/Linux/Windows native)
# 2. Pull a model
ollama pull gemma4:e4b   # 9 GB, M-series friendly. Use :e2b for 7 GB.
# 3. Confirm Ollama is up
curl http://localhost:11434/api/tags
```

If you (the IDE agent) detect that the user has not done this, give
them these three commands and pause — do not silently fall back to the
mirror baseline and pretend the LLM is firing.

## Verifying the LLM actually ran

After `scm_bench run-scenario`, check the metrics:

```bash
scm_bench metrics --batch-id dev --out runs
```

The `tokens` column reports total tokens generated. If it is `0`, the
fallback fired on every tick — the LLM call is failing. Diagnose with:

```bash
ollama list                                        # is the model pulled?
curl http://localhost:11434/api/tags               # is the server up?
SCB_LLM_TIMEOUT_S=180 scm_bench test-bundle .   # is it just slow?
```

## What to edit

Two things per `agent.py`:

1. **`SYSTEM_PROMPT`** — the per-role brief at the top of the file.
   Default is short and role-specific. Sharpen it. Examples:
   - "Use a base-stock policy with target = mean(last 8 demands) × 3"
   - "If demand has been stable for 5+ periods, hold target constant"
   - "When close to the horizon end, taper orders to avoid stranded inventory"

2. **`step()`** body — if you want anything beyond "ask the LLM and
   return its order." Examples:
   - Pre-process the observation (clip outliers, smooth a window)
   - Post-process the LLM's order with a hard clamp `[0, 50]`
   - Mix LLM + base-stock: trust the LLM during transients, fall back
     to a closed-form rule on stable demand
   - Cache responses across consecutive identical observations (cheap
     speedup if your policy is mostly stationary)

The mirror-fallback contract is the safety net — code defensively
around it.

## Hard constraints (same as Mirror — re-read)

- No cross-agent globals.
- No reading scenario files / engine state.
- `supports_tools:` and `supports_messages:` must match code.
- `tokens_used` reporting is honest — the runtime cross-checks against
  per-tier rate limits when configured.

## When the user says "just run it"

The autopilot path:

```bash
ollama pull gemma4:e4b                           # if not already pulled
scm_bench test-bundle .                    # MUST pass
scm_bench run-scenario --bundle . --scenario s1.1 \
    --out runs --batch-id dev --team-id $(jq -r .team_id manifest.json)
scm_bench metrics --batch-id dev --out runs
```

If `tokens` in the metrics output is 0, the LLM didn't fire — diagnose
before claiming the run worked.
