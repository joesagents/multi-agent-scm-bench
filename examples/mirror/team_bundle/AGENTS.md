# AGENTS.md — scm_bench team bundle (Mirror starter)

Contract for coding agents (Claude Code, Codex, Cursor, etc.).
Read this end-to-end before editing files in this bundle. The user
is the human in the loop; you are the coding agent working on this bundle.

## What this bundle is

A 4-tier supply-chain agent submission. Four `agent.py` files, one per
role. Each implements a `step(observation, inbox, t) -> AgentDecision`
that decides how many units to order this period. The bench harness
runs the four agents in lockstep against a demand scenario; lower
total cost wins.

## Layout

```
team_bundle/
├── manifest.json                 # team_id, sdk_version, agent dirs (edit team_id ONCE)
├── retailer/{agent.py,agent.yaml}
├── wholesaler/{agent.py,agent.yaml}
├── distributor/{agent.py,agent.yaml}
├── factory/{agent.py,agent.yaml}
└── tests/test_local.py           # runs the validator + 5-tick smoke
```

Each `<role>/agent.py` defines exactly one class — `RetailerAgent`,
`WholesalerAgent`, `DistributorAgent`, or `FactoryAgent` — subclassing
`scm_bench.sdk.Agent`. Edit only the `step()` body and any
helper methods on the same class. Do not rename the class, the file,
or the directory.

## How to run the simulation

From the bundle root (the directory containing `manifest.json`):

```bash
# 1. Install deps (one-time per env)
pip install -e .

# 2. Validate + smoke (5 ticks, fast). Run after every edit.
scm-bench test-bundle .

# 3. Full local run on Level 1 stable demand (365 ticks)
scm-bench run-scenario --bundle . --scenario s1.1 \
    --out runs --batch-id dev --team-id $(jq -r .team_id manifest.json)

# 4. Read the metrics back
scm-bench metrics --batch-id dev --out runs
```

Available scenarios in this build: `intro_step_demand` (5-tick smoke),
`s1.1` (Level 1, stable demand, 365 ticks), `s2.3` (Level 2, step shock, 365 ticks).

## How `step()` is called

```python
def step(self,
         observation: LocalObservation,
         inbox: list[Message],
         t: int) -> AgentDecision:
    return AgentDecision(order_qty=...)   # int >= 0
```

`LocalObservation` fields (read-only, all you have):

- `timestep: int` — current period
- `role: str` — your role
- `inventory_on_hand: int`
- `backlog: int`
- `incoming_order_qty: int` — what your downstream tier just ordered
- `incoming_shipment_qty: int` — what your upstream is delivering now
- `pipeline_inventory: int` — units already ordered, not yet arrived
- `order_history_window: list[int]` — recent incoming orders (≤ 8)
- `shipment_history_window: list[int]` — recent deliveries (≤ 8)
- `costs_to_date: CostBreakdown` — `.total`, `.holding`, `.backlog`

`AgentDecision` fields:

- `order_qty: int` (required, ≥ 0)
- `tokens_used: int` (optional; report if you call an LLM)
- `messages: list[Message]` (Phase 2; ignore for now)

## Hard constraints — do NOT violate

The validator and runtime enforce these. Violating them = bundle
rejected by the validator.

- **No global / cross-agent state.** Each `agent.py` is loaded as an
  isolated module. Module-level mutable globals to share between tiers
  are forbidden and won't survive validation anyway (separate processes).
- **No filesystem reads** beyond what `import` already does. No
  reading scenario files, no checkpointing, no logs. Use the agent's
  own `__init__` / `reset` for state.
- **No network calls** unless they're an LLM call (see LLM section
  below) and you've declared the dependency in your write-up. Even
  then, the runner may execute without network access — your agent must
  produce reasonable orders on the fallback path.
- **No imports from `scm_bench.scenarios.*` or
  `scm_bench.engine.*`.** Those let you peek at the demand
  function or other tiers' state — that's an automatic disqualification.
- **`agent.yaml` `supports_tools` and `supports_messages` must match
  what your code actually does.** The Phase 2 enforcer will reject
  bundles that declare-and-don't-use or use-and-don't-declare.

## Memory (if you need it)

Default `memory_mode: stateless` in `agent.yaml`. If your policy needs
to remember anything across ticks, change it to `bounded_buffer` and
set `memory_max_entries` (1..4096). Then store on `self`:

```python
class RetailerAgent(Agent):
    def reset(self, *, role, config, seed):
        super().reset(role=role, config=config, seed=seed)
        self._demand_history: list[int] = []

    def step(self, observation, inbox, t):
        self._demand_history.append(observation.incoming_order_qty)
        if len(self._demand_history) > 32:
            self._demand_history.pop(0)
        ...
```

The validator clamps to the declared max — if you grow past it, the
oldest entries silently drop. There is no penalty for using less.

## Built-in tools (optional)

Three pure-Python helpers live in
`scm_bench.tools.builtin`:

- `forecast_moving_average(values, window=4) -> float`
- `forecast_exponential_smoothing(values, alpha=0.3) -> float`
- `local_cost_estimator(inventory, backlog, holding=0.50, backlog=1.00) -> float`

If you call any of them, list them in `supports_tools:` in
`agent.yaml`. Phase 2 will enforce match-up.

## LLM agents (optional path)

If you want an LLM to drive `step()`, the wheel ships a helper:

```python
from scm_bench.starters.llm_starter import decide_with_llm, make_runtime
```

Default backend: Ollama at `http://localhost:11434/v1`, model
`gemma4:e4b`. To use it: install Ollama, `ollama pull gemma4:e4b`,
restart your shell. The helper falls back to the mirror policy on any
LLM error so the bundle still validates without Ollama running.

A complete LLM-driven starter bundle ships separately as
`starter_bundle_llm.zip` — unzip that into a fresh directory if you
want to start from there instead of from this Mirror bundle.

Env vars:

| Variable | Default |
|---|---|
| `SCB_LLM_BACKEND` | `ollama` |
| `SCB_LLM_MODEL` | `gemma4:e4b` |
| `SCB_LLM_BASE_URL` | `http://localhost:11434/v1` |
| `SCB_LLM_API_KEY` | unset |
| `SCB_LLM_TIMEOUT_S` | 60 |

## Edit-test loop

After any change:

```bash
scm-bench test-bundle .         # MUST pass before anything else
scm-bench run-scenario --bundle . --scenario s1.1 \
    --out runs --batch-id dev --team-id <team>
scm-bench metrics --batch-id dev --out runs
```

The `composite` column is the score. **Lower is better.** Mirror
baseline = 1.000 on all scenarios; beat it.

## Common errors

| Error | Likely cause |
|---|---|
| `E_AGENT_MANIFEST_SCHEMA` | You changed `agent.yaml` to an invalid value. |
| `E_AGENT_ROLE_MISMATCH` | `role:` in yaml doesn't match the directory name. |
| `entrypoint class not found` | Class name in `entrypoint:` doesn't exist in `agent.py`. |
| `ModuleNotFoundError: scm_bench` | Wheel not installed in the active env. |
| Smoke hangs at 5 ticks | `step()` raised or returned `None`. Must return `AgentDecision`. |
| Composite > 10 | Bundle is amplifying bullwhip. Smooth your orders. |

## Submitting

Zip the bundle root **flat** (the contents of `team_bundle/`, not the
directory itself):

```bash
cd /path/to/team_bundle
zip -r ../my_team_submission.zip . -x '*.pyc' '__pycache__/*'
```

Then archive or share the bundle as needed. Don't include the
`runs/` output directory; downstream re-runs regenerate it.
