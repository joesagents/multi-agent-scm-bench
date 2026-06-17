---
name: run
description: Drive multi-agent-scm-bench — validate a team bundle, run it on a scenario (or the full scenario x seed matrix), and read the composite verdict (lower is better). Use when running the bench, scoring a bundle, or starting a new team. The engine, validator, and metrics decide the result; this skill operates them, it does not tune or override them.
---

# Run

Operate the bench: feed it a team bundle, run it on a scenario, read the
verdict. The score is the `composite` column and **lower is better**; the
mirror baseline is `1.000`, so a team is doing useful work when it scores below
that.

Run from declared inputs and read the emitted verdict. Do not edit scenarios,
metrics, gates, or the engine to make a run look better — those define what the
score means.

## Validate a bundle

A bundle is a directory containing `manifest.json` and four role subdirectories
(`retailer`, `wholesaler`, `distributor`, `factory`). Validate it first:

```bash
scm-bench test-bundle <bundle-dir>        # schema + entrypoints + 5-tick smoke
```

By default the bundle runs in an isolated subprocess with wall-clock and memory
caps. Pass `--in-process` only for a bundle you trust (e.g. your own during
development), and `--json` for a machine-readable report.

## Run one scenario

```bash
scm-bench run-scenario --bundle <bundle-dir> --scenario s1.1 \
    --out runs --batch-id dev --team-id <team-id>
scm-bench metrics --batch-id dev --out runs
```

Built-in scenario ids: `intro_step_demand` (5-tick smoke), `s1.1` (stable
demand, 365 ticks), `s2.3` (step shock), `s2.4` (seeded stochastic shock). The
RunStore (`--out runs`) holds `runs.db` plus a JSONL tree; re-runs regenerate
it, so it does not need to be committed.

Use `--variant <tag> --override role.config.key=value` to run the same bundle
with a tweaked per-role config and group the sibling rows in reports.

## Run the full matrix

```bash
scm-bench batch-run --submissions <dir-of-bundles> \
    --scenarios s1.1,s2.3 --seeds 0,1,2 --batch-id <id> --out runs
scm-bench report --batch-id <id> --runs runs --out reports
```

`batch-run` runs every bundle subdirectory in `--submissions` across the
scenario x seed grid; reference starter teams are included by default
(`--no-include-starters` to drop them). `report` writes per-team Markdown
rundowns plus one aggregate summary.

## Start a new team

```bash
scm-bench export-template -o team_bundle    # copy the starter template
# edit team_bundle/<role>/agent.py, then validate:
scm-bench test-bundle team_bundle
```

Edit only the `step()` body and helper methods on the role class; do not rename
the class, file, or directory. The per-bundle `AGENTS.md` is the authoritative
contract for what an agent may and may not do (no cross-agent state, no peeking
at `scm_bench.scenarios` / `scm_bench.engine`, declared tools/messages must
match the code).
