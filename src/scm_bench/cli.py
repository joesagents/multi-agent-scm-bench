"""scm_bench CLI.

Default + bundle-side commands:
- scm_bench                 : default local run on the mirror starter team
- scm_bench test-bundle PATH: validate a bundle and run a smoke (sandboxed
                                      subprocess by default)
- scm_bench export-template : copy the starter template into a new dir

Batch-run + reporting commands (P2a, P2b):
- scm_bench run-scenario    : one bundle × one scenario × one seed → RunStore
- scm_bench batch-run       : full team × scenario × seed matrix
- scm_bench metrics         : print summary metrics for a batch-run
- scm_bench report          : per-team Markdown rundowns + aggregate summary

Phase 3 stubs: replay, leaderboard, compare.

A `beergame` deprecation alias entrypoint forwards to the same Typer
app and prints a one-line stderr warning. It will be removed in the
next release.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import typer

BUNDLE_MARKER_FILENAME = ".scm_bench-bundle"
LEGACY_BUNDLE_MARKER_FILENAME = ".beergame-bundle"

from scm_bench import SDK_VERSION
from scm_bench.runner.batch import (
    TeamSpec,
    batch_run as batch_orchestrator,
    reference_starter_specs,
)
from scm_bench.runner.harness import run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.sdk.validator import (
    DEFAULT_SANDBOX_TIMEOUT_S,
    ValidationError,
    validate_bundle,
    validate_bundle_isolated,
    validate_bundle_safe,
)
from scm_bench.starters.mirror import MirrorAgent
from scm_bench.trace.store import MIRROR_TEAM_ID, RunStore

app = typer.Typer(
    name="scm_bench",
    help="Supply chain bench — multi-agent benchmark (Phase 1 commands).",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Default command: run the mirror starter team on intro_step_demand."""
    if ctx.invoked_subcommand is not None:
        return
    scenario = scenarios_builtin.INTRO_STEP_DEMAND
    agents = {
        "retailer": MirrorAgent(),
        "wholesaler": MirrorAgent(),
        "distributor": MirrorAgent(),
        "factory": MirrorAgent(),
    }
    record = run_one(
        agents=agents,
        scenario=scenario,
        seed=0,
        run_id="default-mirror",
    )
    result = record.result
    assert result is not None
    typer.echo(f"scenario       : {scenario.id}")
    typer.echo(f"horizon        : {result.periods_run}")
    typer.echo(f"total cost     : {result.total_cost:.2f}")
    typer.echo(f"bullwhip       : {result.bullwhip:.3f}")
    typer.echo(f"chain fill     : {result.chain_fill_rate:.3f}")
    typer.echo(f"avg inventory  : {result.average_inventory:.2f}")
    typer.echo(f"avg backlog    : {result.average_backlog:.2f}")
    for tier_name, tier_result in result.tier_results.items():
        typer.echo(
            f"  {tier_name:12s} cost={tier_result.total_cost:.2f}  "
            f"fill={tier_result.fill_rate:.3f}"
        )


@app.command("test-bundle")
def test_bundle(
    bundle_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Path to the team bundle directory (containing manifest.json).",
    ),
    smoke_ticks: int = typer.Option(
        5, "--smoke-ticks", help="Number of ticks for the smoke run."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a machine-readable JSON report."
    ),
    in_process: bool = typer.Option(
        False,
        "--in-process",
        help="Skip subprocess isolation. Only use on bundles you trust "
        "(e.g. your own during development).",
    ),
    timeout_s: float = typer.Option(
        DEFAULT_SANDBOX_TIMEOUT_S,
        "--timeout",
        help="Wall-clock budget for the isolated worker (seconds).",
    ),
) -> None:
    """Validate a bundle: schema, entrypoints, smoke run.

    By default the bundle is imported and run inside a subprocess with
    a wall-clock timeout and (POSIX) RLIMIT_AS / RLIMIT_CPU caps. A
    timeout surfaces as E_BUNDLE_TIMEOUT and a crash as E_BUNDLE_CRASH.
    Pass --in-process to run in this Python process instead (no
    isolation; use only for bundles you trust).
    """
    if in_process:
        if json_output:
            report = validate_bundle_safe(bundle_path, smoke_ticks=smoke_ticks)
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
            sys.exit(0 if report["ok"] else 1)

        try:
            report = validate_bundle(bundle_path, smoke_ticks=smoke_ticks)
        except ValidationError as e:
            typer.secho(f"FAIL [{e.code}] {e}", fg=typer.colors.RED, err=True)
            sys.exit(1)
        typer.secho("OK", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  team_id      : {report.team.team_id}")
        typer.echo(f"  team_name    : {report.team.team_name}")
        typer.echo(f"  sdk_version  : {report.team.sdk_version}")
        for role, am in report.agents.items():
            typer.echo(
                f"  {role:12s} {am.agent_name}  "
                f"memory={am.memory_mode}  tools={am.supports_tools}  "
                f"messages={[m.value for m in am.supports_messages]}"
            )
        if report.smoke_run_record is not None:
            typer.echo(f"  smoke ticks  : {len(report.smoke_run_record.ticks)}")
        return

    payload = validate_bundle_isolated(
        bundle_path, smoke_ticks=smoke_ticks, timeout_s=timeout_s
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        sys.exit(0 if payload.get("ok") else 1)
    if not payload.get("ok"):
        typer.secho(
            f"FAIL [{payload.get('code')}] {payload.get('message')}",
            fg=typer.colors.RED,
            err=True,
        )
        sys.exit(1)
    typer.secho("OK", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  team_id      : {payload['team_id']}")
    typer.echo(f"  team_name    : {payload['team_name']}")
    typer.echo(f"  sdk_version  : {payload['sdk_version']}")
    for role, am in payload["agents"].items():
        typer.echo(
            f"  {role:12s} {am['agent_name']}  "
            f"memory={am['memory_mode']}  tools={am['supports_tools']}  "
            f"messages={am['supports_messages']}"
        )
    typer.echo(f"  smoke ticks  : {payload['smoke_ticks']}")


@app.command("export-template")
def export_template(
    out: Path = typer.Option(
        Path("./team_bundle"),
        "--out",
        "-o",
        help="Destination directory (must not exist).",
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing destination."
    ),
) -> None:
    """Copy the starter template into a fresh team_bundle directory."""
    if out.exists():
        if not force:
            typer.secho(
                f"destination {out} already exists; use --force to overwrite",
                fg=typer.colors.RED,
                err=True,
            )
            sys.exit(1)
        # Refuse to rmtree anything that wasn't created by export-template.
        # The marker is dropped at the end of a successful copy below; an
        # earlier failed copy may leave a partial dir without the marker, so
        # we also accept a recognisable team manifest.json as evidence that
        # this is a bundle directory and not unrelated user data.
        if not _is_safe_to_overwrite(out):
            typer.secho(
                f"refusing --force overwrite of {out}: not a scm_bench "
                f"bundle (no {BUNDLE_MARKER_FILENAME} marker and no "
                "manifest.json). Move or delete the directory by hand if you "
                "really mean it.",
                fg=typer.colors.RED,
                err=True,
            )
            sys.exit(1)
        shutil.rmtree(out)

    template_root = resources.files("scm_bench.sdk").joinpath(
        "starter_template"
    )
    with resources.as_file(template_root) as template_path:
        shutil.copytree(template_path, out)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sdk_version"] = SDK_VERSION
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for role_dir in ("retailer", "wholesaler", "distributor", "factory"):
        agent_yaml = out / role_dir / "agent.yaml"
        text = agent_yaml.read_text()
        text = text.replace('sdk_version: "1.0.0"', f'sdk_version: "{SDK_VERSION}"')
        agent_yaml.write_text(text)

    _write_bundle_marker(out)

    typer.secho(f"OK  starter template copied to {out}", fg=typer.colors.GREEN)
    typer.echo(f"  edit  : {out}/<role>/agent.py")
    typer.echo(f"  test  : scm_bench test-bundle {out}")


def _write_bundle_marker(bundle_root: Path) -> None:
    marker = {
        "created_by": "scm_bench export-template",
        "sdk_version": SDK_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (bundle_root / BUNDLE_MARKER_FILENAME).write_text(
        json.dumps(marker, indent=2) + "\n"
    )


def _is_safe_to_overwrite(path: Path) -> bool:
    """True iff `path` is recognisable as a scm_bench bundle directory."""
    if not path.is_dir():
        return False
    # Accept either the new marker or the legacy `.beergame-bundle` marker so
    # that bundle dirs created with the previous CLI name are still
    # recognised by `--force`. (One-release back-compat shim — drop alongside
    # the `beergame` deprecation alias next release.)
    if (path / BUNDLE_MARKER_FILENAME).exists():
        return True
    if (path / LEGACY_BUNDLE_MARKER_FILENAME).exists():
        return True
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "team_id" in data and "sdk_version" in data


def _parse_int_csv(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def _parse_str_csv(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@app.command("run-scenario")
def run_scenario(
    bundle: Path = typer.Option(
        ..., "--bundle",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True,
        help="Path to a validated bundle directory.",
    ),
    scenario: str = typer.Option(..., "--scenario", help="Scenario id."),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(
        Path("./runs"), "--out", "-o",
        help="RunStore root (creates runs.db + JSONL tree).",
        resolve_path=True,
    ),
    batch_id: str = typer.Option(
        "ad-hoc", "--batch-id",
        help="Tag this run with a batch_id for later aggregation.",
    ),
    team_id: str | None = typer.Option(
        None, "--team-id",
        help="Override the team_id (defaults to the bundle's team manifest id).",
    ),
    variant: str = typer.Option(
        "",
        "--variant",
        help="Variant tag for the run (e.g. 'try_q60'). Recorded in runs.db "
        "so the per-team sensitivity table can group sibling runs.",
    ),
    override: list[str] | None = typer.Option(
        None,
        "--override",
        help="Per-role config override: 'role.config.key=value', repeatable.",
    ),
) -> None:
    """Run one bundle on one scenario × one seed; persist to a RunStore.

    `--variant try_q60 --override retailer.config.target_stock=60` runs the
    same bundle with a tweaked per-role config and tags the row with the
    variant id so report builders can show baseline vs. variant side by side.
    """
    sc = scenarios_builtin.get(scenario)
    spec_team_id = team_id or _team_id_from_bundle(bundle)
    overrides = _parse_overrides(override or [])
    spec = TeamSpec(
        team_id=spec_team_id,
        bundle_path=bundle,
        variant_id=variant,
        overrides=overrides,
    )
    store = RunStore(out)
    summary = batch_orchestrator(
        batch_id=batch_id,
        teams=[spec],
        scenario_ids=[sc.id],
        seeds=[seed],
        store=store,
        include_mirror=True,
    )
    _print_class_summary(summary)


def _parse_overrides(items: list[str]) -> dict[str, dict[str, object]]:
    """Parse `role.config.key=value` items into per-role agent config dicts.

    The harness passes `agent_configs[role]` as `config=` to `agent.reset()`,
    so the syntax `role.config.key=value` flattens into `{role: {key: value}}`.
    Only that shape is accepted — other shapes are rejected so a typo
    doesn't silently no-op. Values are coerced int → float → str.
    """
    out: dict[str, dict[str, object]] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(
                f"--override {raw!r} must be of the form role.config.key=value"
            )
        path, value = raw.split("=", 1)
        parts = path.split(".")
        if len(parts) != 3 or parts[1] != "config":
            raise typer.BadParameter(
                f"--override {raw!r}: only role.config.key=value is supported"
            )
        role, _, key = parts
        coerced: object
        try:
            coerced = int(value)
        except ValueError:
            try:
                coerced = float(value)
            except ValueError:
                coerced = value
        out.setdefault(role, {})[key] = coerced
    return out


@app.command("batch-run")
def batch_run(
    submissions: Path | None = typer.Option(
        None, "--submissions",
        exists=False, resolve_path=True,
        help="Directory of bundle subdirectories.",
    ),
    scenarios: str = typer.Option(
        "s1.1,s2.3", "--scenarios",
        help="Comma-separated scenario ids.",
    ),
    seeds: str = typer.Option(
        "0,1,2", "--seeds",
        help="Comma-separated integer seeds.",
    ),
    out: Path = typer.Option(
        Path("./runs"), "--out", "-o",
        help="RunStore root (creates runs.db + JSONL tree).",
        resolve_path=True,
    ),
    batch_id: str = typer.Option(
        ..., "--batch-id",
        help="Identifier for this batch-run; appears in the report path.",
    ),
    include_starters: bool = typer.Option(
        True, "--include-starters/--no-include-starters",
        help="Add Mirror/MovingAverage/BaseStock/CommunicatingForecast as anchor teams.",
    ),
) -> None:
    """Run all bundles in a directory across scenarios × seeds."""
    scenario_ids = _parse_str_csv(scenarios)
    seed_list = _parse_int_csv(seeds)

    teams: list[TeamSpec] = []
    if submissions is not None:
        if not submissions.exists() or not submissions.is_dir():
            typer.secho(
                f"--submissions {submissions} is not a directory",
                fg=typer.colors.RED, err=True,
            )
            sys.exit(1)
        for child in sorted(submissions.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "manifest.json").exists():
                continue
            teams.append(
                TeamSpec(
                    team_id=_team_id_from_bundle(child),
                    bundle_path=child,
                )
            )

    if include_starters:
        teams.extend(reference_starter_specs())

    if not teams:
        typer.secho(
            "no teams to run — pass --submissions <dir> and/or --include-starters",
            fg=typer.colors.RED, err=True,
        )
        sys.exit(1)

    store = RunStore(out)
    summary = batch_orchestrator(
        batch_id=batch_id,
        teams=teams,
        scenario_ids=scenario_ids,
        seeds=seed_list,
        store=store,
        include_mirror=True,
    )
    _print_class_summary(summary)


@app.command("metrics")
def metrics_cmd(
    batch_id: str = typer.Option(..., "--batch-id"),
    out: Path = typer.Option(
        Path("./runs"), "--out", "-o",
        help="RunStore root.",
        resolve_path=True,
    ),
) -> None:
    """Print summary metrics for every cell in a batch-run."""
    store = RunStore(out)
    rows = store.list_runs(batch_id=batch_id)
    if not rows:
        typer.secho(
            f"no runs found for batch_id={batch_id!r} in {out}",
            fg=typer.colors.RED, err=True,
        )
        sys.exit(1)
    typer.echo(
        f"{'team':32s} {'scenario':10s} {'seed':>4s} {'cost':>10s} "
        f"{'bw_ratio':>8s} {'fill':>6s} {'tokens':>8s} {'composite':>10s}"
    )
    for r in rows:
        composite = "—" if r.composite_score is None else f"{r.composite_score:.3f}"
        typer.echo(
            f"{r.team_id:32s} {r.scenario_id:10s} {r.seed:>4d} "
            f"{(r.total_cost or 0.0):>10.2f} "
            f"{(r.bullwhip_ratio or 0.0):>8.3f} "
            f"{(r.chain_fill_rate or 0.0):>6.3f} "
            f"{(r.tokens_used or 0):>8d} {composite:>10s}"
        )


def _team_id_from_bundle(bundle_path: Path) -> str:
    """Cheap manifest-only read; full validation happens in batch-run."""
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            tid = data.get("team_id")
            if isinstance(tid, str) and tid:
                return tid
        except json.JSONDecodeError:
            pass
    return bundle_path.name


def _print_class_summary(summary) -> None:  # noqa: ANN001 — local helper
    ok = sum(1 for c in summary.cells if c.ok)
    fail = len(summary.cells) - ok
    typer.echo(f"batch_id : {summary.batch_id}")
    typer.echo(f"cells        : {len(summary.cells)} ({ok} ok, {fail} failed)")
    if summary.skipped_teams:
        typer.echo(f"skipped teams: {len(summary.skipped_teams)}")
        for team_id, err in summary.skipped_teams:
            typer.echo(f"  {team_id:32s} {err}")
    for c in summary.cells:
        if c.ok:
            assert c.row is not None
            cs = "—" if c.row.composite_score is None else f"{c.row.composite_score:.3f}"
            typer.echo(
                f"  ok   {c.team_id:32s} {c.scenario_id:10s} seed={c.seed:<2d} "
                f"composite={cs}"
            )
        else:
            typer.echo(
                f"  FAIL {c.team_id:32s} {c.scenario_id:10s} seed={c.seed:<2d} "
                f"err={c.error}"
            )


@app.command("report")
def report_cmd(
    batch_id: str = typer.Option(
        ..., "--batch-id",
        help="The batch-run identifier whose runs should be reported on.",
    ),
    out: Path = typer.Option(
        Path("./reports"), "--out", "-o",
        help="Report root (per-team rundowns + aggregate summary land here).",
        resolve_path=True,
    ),
    runs: Path = typer.Option(
        Path("./runs"), "--runs",
        help="RunStore root (where runs.db lives).",
        resolve_path=True,
    ),
    teams_only: str | None = typer.Option(
        None, "--teams-only",
        help="Comma-separated subset of team_ids; default is every team in the run.",
    ),
    git_sha: str | None = typer.Option(
        None, "--git-sha",
        help="Git SHA to embed in the reproducibility footer.",
    ),
    skip_baselines: bool = typer.Option(
        True, "--skip-baselines/--include-baselines",
        help="Skip rendering rundowns for the Mirror baseline and starter teams.",
    ),
) -> None:
    """Render per-team Markdown rundowns + a single aggregate summary."""
    from scm_bench.reporting.aggregate import write_aggregate_summary
    from scm_bench.reporting.per_team import (
        team_ids_in_batch,
        write_team_rundown,
    )
    from scm_bench.trace.store import MIRROR_TEAM_ID

    store = RunStore(runs)
    if not store.list_runs(batch_id=batch_id):
        typer.secho(
            f"no runs found for batch_id={batch_id!r} in {runs}",
            fg=typer.colors.RED, err=True,
        )
        sys.exit(1)

    if teams_only:
        team_ids = _parse_str_csv(teams_only)
    else:
        exclude: set[str] = set()
        if skip_baselines:
            exclude = {MIRROR_TEAM_ID} | {
                r.team_id for r in store.list_runs(batch_id=batch_id)
                if r.team_id.startswith("__starter_") or r.team_id.startswith("__llm_")
            }
        team_ids = team_ids_in_batch(
            store=store, batch_id=batch_id, exclude=exclude
        )

    if not team_ids:
        typer.secho(
            "no teams to report on — pass --include-baselines or --teams-only",
            fg=typer.colors.RED, err=True,
        )
        sys.exit(1)

    for team_id in team_ids:
        path = write_team_rundown(
            store=store,
            batch_id=batch_id,
            team_id=team_id,
            out_root=out,
            git_sha=git_sha,
        )
        typer.echo(f"wrote {path}")

    summary_path = write_aggregate_summary(
        store=store,
        batch_id=batch_id,
        out_root=out,
        git_sha=git_sha,
    )
    typer.echo(f"wrote {summary_path}")


@app.command("replay")
def replay_cmd(
    run_id: str = typer.Argument(...),
    tick: int | None = typer.Option(None, "--tick"),
) -> None:
    """[Phase 3] Replay a recorded run, optionally at a specific tick."""
    raise typer.Exit(typer.echo("replay is not yet shipped (Phase 3).", err=True) or 2)


@app.command("leaderboard")
def leaderboard_cmd(scenario: str = typer.Argument(...)) -> None:
    """[Phase 3] Rank bundles for a scenario."""
    raise typer.Exit(
        typer.echo("leaderboard is not yet shipped (Phase 3).", err=True) or 2
    )


@app.command("compare")
def compare_cmd(
    run_id_1: str = typer.Argument(...),
    run_id_2: str = typer.Argument(...),
) -> None:
    """[Phase 3] Side-by-side comparison of two runs."""
    raise typer.Exit(typer.echo("compare is not yet shipped (Phase 3).", err=True) or 2)



def beergame_deprecated() -> None:
    """Deprecation shim for the old `beergame` console_scripts entry point.

    Forwards every invocation to the canonical `scm_bench` Typer
    app after printing a one-line stderr warning. The `beergame`
    entrypoint will be removed in the next release.
    """
    print(
        "warning: `beergame` is deprecated and will be removed next release; "
        "use `scm_bench` instead.",
        file=sys.stderr,
    )
    app()


if __name__ == "__main__":
    app()
