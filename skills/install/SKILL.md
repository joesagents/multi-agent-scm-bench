---
name: install
description: Install multi-agent-scm-bench locally and verify the checkout with the default run, the bundle smoke, and pytest. Use before running scenarios, validating a team bundle, or driving an LLM agent. NOT for modifying the bench internals.
---

# Install

Install the `scm_bench` package into a fresh virtual environment and prove it
works. When the default run prints a cost line and `test-bundle` on the mirror
example is green, the bench is trustworthy — switch to the `run` skill to drive
it. Default to a local install; do not reach for cluster resources for setup
unless the user explicitly asks for a cluster run.

## Local install

From the repository root (the directory containing `pyproject.toml`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Use an interpreter that satisfies `requires-python` in `pyproject.toml`.
`pip install -e .` pulls the runtime dependencies declared there under
`[project] dependencies` — no LLM or solver backend is needed for the core
bench, since the reference teams use deterministic policies.

## Verify

Run all three; each should succeed before you move on:

```bash
scm-bench                                   # default run: prints scenario/cost/bullwhip/fill
scm-bench test-bundle examples/mirror/team_bundle   # validate + 5-tick smoke -> "OK"
pip install -e '.[dev]' && pytest           # full suite
```

A green `test-bundle` ("OK" plus a per-role table and `smoke ticks : 5`) means
the apparatus is sound. If `scm-bench` is not found, the editable install did
not put the script on PATH — confirm the venv is activated and re-run
`pip install -e .`. `ModuleNotFoundError: scm_bench` means the wheel is not
installed in the active env.

## LLM agents (optional)

Only needed if you drive `step()` with a model. The bench falls back to the
mirror policy on any LLM error, so install and the core flow work without this.

```bash
pip install -e '.[llm-transformers]'
pip install -e '.[llm-vllm]'
```

The available extras and what each pulls in are declared under
`[project.optional-dependencies]` in `pyproject.toml`. The LLM agent talks to an
OpenAI-compatible endpoint (a local Ollama by default) and is configured through
the `SCB_LLM_*` environment variables — see each `examples/<name>/AGENTS.md` for
the full LLM contract.

Building or changing the bench internals is out of scope; this skill installs
and verifies it, it does not modify it.
