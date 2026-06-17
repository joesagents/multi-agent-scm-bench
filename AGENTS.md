# AGENTS.md — multi-agent-scm-bench

Contract for coding agents (Claude Code, Codex, Cursor, …) working at the repo
root. This is a 4-tier agentic supply-chain ("beer game") benchmark: four
agents — retailer → wholesaler → distributor → factory — each decide an order
quantity per period, run in lockstep against a demand scenario. Lower total cost
wins.

You will usually be asked to do one of two things: **operate** the bench
(install + run it), or **build a team** that competes in it.

## Install & verify

From the repo root (Python ≥ 3.11 — the floor is `requires-python` in
`pyproject.toml`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
scm-bench                                            # default run prints a cost line
scm-bench test-bundle examples/mirror/team_bundle    # -> OK
```

`pip install -e '.[dev]' && pytest` runs the full suite. The installed command is
`scm-bench`. No LLM or solver backend is needed for the core bench. Full detail:
`skills/install`.

## Operate the bench

Validate a bundle, run it on a scenario, read the verdict — the `composite`
column, **lower is better**; the Mirror baseline is `1.000`:

```bash
scm-bench test-bundle <bundle>
scm-bench run-scenario --bundle <bundle> --scenario s1.1 \
    --out runs --batch-id dev --team-id <id>
scm-bench metrics --batch-id dev --out runs
```

Scenarios: `s1.1`, `s2.3`, `s2.4` (plus `intro_step_demand` for a 5-tick smoke).
Full command reference: `skills/run`.

## Build a team (add your agents)

```bash
scm-bench export-template -o my_team
```

This writes a bundle with four `agent.py` stubs and **its own `AGENTS.md`** —
that per-bundle `AGENTS.md` is the authoritative contract for writing agents: the
`step()` signature, the read-only observation fields, the hard constraints, the
optional LLM path (with a fallback so the bundle still validates without a live
model), and how to submit. Open `my_team/` and follow it. Edit only the `step()`
body of each `<role>/agent.py`, and validate after every change:

```bash
scm-bench test-bundle my_team
```

`examples/mirror/` (deterministic baseline) and `examples/llm/` (LLM-driven) are
worked bundles to copy from.

Directory layout: README.md. License 0BSD; tools and hardware:
  ATTRIBUTION.md.
