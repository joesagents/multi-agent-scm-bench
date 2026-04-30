"""Batch-run orchestrator — teams × scenarios × seeds × variants.

The operator-facing entry point. For each (bundle, scenario, seed,
variant) cell:

1. Resolve agents: either from a validated bundle (submitted team) or
   from a built-in starter factory (Mirror as the echo baseline plus
   the three reference policies).
2. Run the cell via `runner.harness.run_one`.
3. Persist via `trace.store.RunStore`.
4. After all cells in the matrix complete, normalise composite scores
   against the Mirror baseline cell (same scenario × seed) and write
   them back into the SQLite index.

Composite scoring rule, from the metrics primer:

    composite_score = geometric_mean(
        bullwhip_ratio_norm, total_cost_norm, tokens_used_norm
    )

where each `_norm = (your + ε) / (mirror + ε)` with ε = 1.0.

The Mirror baseline is therefore mandatory: every batch-run injects it
under team_id `__mirror_baseline__` (constant in `trace.store`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scm_bench import SDK_VERSION
from scm_bench.metrics.composite import (
    EPSILON,
    ScoreComponents,
    normalize_components,
)
from scm_bench.runner.harness import PreTickCallback, RunRecord, run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.scenarios.scenario import Scenario
from scm_bench.sdk.agent import Agent
from scm_bench.sdk.validator import (
    ValidationError,
    ValidationReport,
    validate_bundle,
)
from scm_bench.starters.base_stock import BaseStockAgent
from scm_bench.starters.communicating_forecast import (
    CommunicatingForecastAgent,
)
from scm_bench.starters.llm_baseline import make_llm_baseline_team
from scm_bench.starters.llm_coached import make_coached_llm_team
from scm_bench.starters.mirror import MirrorAgent
from scm_bench.starters.moving_average import MovingAverageAgent
from scm_bench.trace.store import MIRROR_TEAM_ID, RunRow, RunStore

AgentFactory = Callable[[], Agent]
TeamBuilder = Callable[[], tuple[dict[str, Agent], PreTickCallback | None]]


REFERENCE_STARTER_FACTORIES: dict[str, AgentFactory] = {
    "__starter_moving_average__": MovingAverageAgent,
    "__starter_base_stock__": BaseStockAgent,
    "__starter_communicating_forecast__": CommunicatingForecastAgent,
}


@dataclass(frozen=True)
class TeamSpec:
    """One participant in a batch-run.

    Exactly one of ``bundle_path``, ``factory``, or ``team_builder``
    must be set:

    - ``bundle_path``: agent bundle, will be validated.
    - ``factory``: built-in starter (called four times, once per role).
    - ``team_builder``: returns ``(agents, pre_tick_callback)`` in one
      shot. Used by the LLM baseline so all four tiers share a single
      `LLMRuntime` and one batched `model.generate` per tick.

    ``team_id`` is the primary key in the run store.
    """

    team_id: str
    bundle_path: Path | None = None
    factory: AgentFactory | None = None
    team_builder: TeamBuilder | None = None
    variant_id: str = ""
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sources = [
            self.bundle_path is not None,
            self.factory is not None,
            self.team_builder is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                f"TeamSpec(team_id={self.team_id!r}) must set exactly one of "
                "bundle_path, factory, or team_builder"
            )


@dataclass(frozen=True)
class CellOutcome:
    team_id: str
    scenario_id: str
    seed: int
    variant_id: str
    row: RunRow | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.row is not None


@dataclass(frozen=True)
class BatchRunSummary:
    batch_id: str
    cells: list[CellOutcome]
    skipped_teams: list[tuple[str, str]]  # (team_id, error)


def mirror_team_spec() -> TeamSpec:
    return TeamSpec(team_id=MIRROR_TEAM_ID, factory=MirrorAgent)


def reference_starter_specs() -> list[TeamSpec]:
    return [
        TeamSpec(team_id=team_id, factory=factory)
        for team_id, factory in REFERENCE_STARTER_FACTORIES.items()
    ]


def llm_baseline_team_spec(
    *,
    model_id: str,
    team_id: str | None = None,
    backend_name: str | None = None,
    max_new_tokens: int = 48,
) -> TeamSpec:
    """Build a TeamSpec that wraps a Gemma model as a 4-tier baseline team.

    The model is loaded lazily inside `team_builder` so import-time
    has no GPU dependency. The same `LLMRuntime` is shared across all
    four tiers; the harness uses the team's `pre_tick_callback` to run
    one batched `model.generate` per tick instead of four sequential
    calls.
    """
    from scm_bench.runner.llm_runtime import LLMRuntime

    tid = team_id or f"__llm_baseline__{model_id.split('/')[-1]}__"

    def _builder() -> tuple[dict[str, Agent], PreTickCallback | None]:
        runtime = LLMRuntime(
            model_id=model_id,
            backend_name=backend_name,
            max_new_tokens=max_new_tokens,
        )
        agents, callback = make_llm_baseline_team(runtime)
        return dict(agents), callback

    _builder.__name__ = f"llm_baseline_builder[{model_id}]"
    return TeamSpec(team_id=tid, team_builder=_builder)


def coached_llm_baseline_team_spec(
    *,
    model_id: str,
    team_id: str | None = None,
    backend_name: str | None = None,
    max_new_tokens: int = 64,
    horizon_hint: int = 365,
) -> TeamSpec:
    """Build a TeamSpec for the *coached* LLM baseline.

    Same shared-runtime / batched-generate pattern as
    `llm_baseline_team_spec`, but the per-tick callback emits a
    structured `{pattern, target, halt, why}` policy spec rather than
    a raw integer order. Code computes the order from the spec via
    base-stock arithmetic; the LLM is the strategist, not the executor.

    `horizon_hint` is the number of ticks the model expects in the
    scenario (used by the prompt to decide when to halt new orders in
    the lead-time tail). Defaults to 365 (the long-horizon default
    used by S1.1/S2.3).
    """
    from scm_bench.runner.llm_runtime import LLMRuntime

    tid = team_id or f"__coached_llm__{model_id.split('/')[-1]}__"

    def _builder() -> tuple[dict[str, Agent], PreTickCallback | None]:
        runtime = LLMRuntime(
            model_id=model_id,
            backend_name=backend_name,
            max_new_tokens=max_new_tokens,
        )
        agents, callback = make_coached_llm_team(
            runtime, horizon_hint=horizon_hint
        )
        return dict(agents), callback

    _builder.__name__ = f"coached_llm_builder[{model_id}]"
    return TeamSpec(team_id=tid, team_builder=_builder)


def _instantiate_team(
    spec: TeamSpec,
) -> tuple[
    dict[str, Agent],
    dict[str, Any],
    dict[str, Any] | None,
    PreTickCallback | None,
]:
    """Resolve a TeamSpec into (agents, bundle_meta, manifests, pre_tick_cb).

    `agent_manifests` is None for built-in factories / team builders
    (no contract to enforce — they are trusted) and a role→AgentManifest
    dict for agent bundles. `pre_tick_callback` is non-None only for
    `team_builder` specs that return one (the LLM baseline).

    Raises ValidationError if the bundle is invalid.
    """
    if spec.factory is not None:
        agents = {
            role: spec.factory()
            for role in ("retailer", "wholesaler", "distributor", "factory")
        }
        meta = {"source": "factory", "factory": spec.factory.__name__}
        return agents, meta, None, None

    if spec.team_builder is not None:
        agents, callback = spec.team_builder()
        meta = {"source": "team_builder", "builder": spec.team_builder.__name__}
        return agents, meta, None, callback

    assert spec.bundle_path is not None
    report: ValidationReport = validate_bundle(spec.bundle_path, smoke_ticks=2)
    agents: dict[str, Agent] = {}
    for role, agent_manifest in report.agents.items():
        role_dir = report.team.agents[role]
        from scm_bench.sdk.validator import _import_entrypoint
        cls = _import_entrypoint(report.bundle_root, role, role_dir, agent_manifest)
        agents[role] = cls()
    meta = {
        "source": "bundle",
        "bundle_path": str(report.bundle_root),
        "team_name": report.team.team_name,
        "submission_version": report.team.submission_version,
    }
    return agents, meta, dict(report.agents), None


def _resolve_scenarios(scenario_ids: list[str]) -> list[Scenario]:
    return [scenarios_builtin.get(sid) for sid in scenario_ids]


def batch_run(
    *,
    batch_id: str,
    teams: list[TeamSpec],
    scenario_ids: list[str],
    seeds: list[int],
    store: RunStore,
    include_mirror: bool = True,
) -> BatchRunSummary:
    """Execute the full team × scenario × seed × variant matrix.

    Cells are run sequentially (single-process). A distributed harness
    can parallelise by indexing into the same matrix.
    """
    matrix_teams = list(teams)
    if include_mirror and not any(t.team_id == MIRROR_TEAM_ID for t in matrix_teams):
        matrix_teams.insert(0, mirror_team_spec())

    scenarios = _resolve_scenarios(scenario_ids)

    cells: list[CellOutcome] = []
    skipped: list[tuple[str, str]] = []

    resolved: dict[
        str,
        tuple[
            dict[str, Agent],
            dict[str, Any],
            dict[str, Any] | None,
            PreTickCallback | None,
        ]
        | None,
    ] = {}
    for spec in matrix_teams:
        try:
            resolved[spec.team_id] = _instantiate_team(spec)
        except ValidationError as e:
            resolved[spec.team_id] = None
            skipped.append((spec.team_id, f"[{e.code}] {e}"))

    for spec in matrix_teams:
        if resolved[spec.team_id] is None:
            continue
        for scenario in scenarios:
            for seed in seeds:
                # Re-instantiate so each cell starts fresh (agents may carry
                # internal state across reset()).
                fresh, bundle_meta, agent_manifests, pre_tick_cb = _instantiate_team(
                    spec
                )
                run_id = f"{batch_id}__{spec.team_id}__{scenario.id}__seed{seed}"
                if spec.variant_id:
                    run_id += f"__{spec.variant_id}"
                try:
                    record: RunRecord = run_one(
                        agents=fresh,
                        scenario=scenario,
                        seed=seed,
                        run_id=run_id,
                        agent_configs=spec.overrides or None,
                        agent_manifests=agent_manifests,
                        pre_tick_callback=pre_tick_cb,
                    )
                    row = store.write_run(
                        record=record,
                        batch_id=batch_id,
                        team_id=spec.team_id,
                        variant_id=spec.variant_id,
                        bundle_meta=bundle_meta,
                        sdk_version=SDK_VERSION,
                    )
                    cells.append(
                        CellOutcome(
                            team_id=spec.team_id,
                            scenario_id=scenario.id,
                            seed=seed,
                            variant_id=spec.variant_id,
                            row=row,
                        )
                    )
                except Exception as e:  # noqa: BLE001 — capture cell-level errors
                    cells.append(
                        CellOutcome(
                            team_id=spec.team_id,
                            scenario_id=scenario.id,
                            seed=seed,
                            variant_id=spec.variant_id,
                            row=None,
                            error=f"{type(e).__name__}: {e}",
                        )
                    )

    if include_mirror:
        _backfill_composite_scores(batch_id=batch_id, store=store)

    refreshed = {
        r.run_id: r for r in store.list_runs(batch_id=batch_id)
    }
    refreshed_cells = [
        CellOutcome(
            team_id=c.team_id,
            scenario_id=c.scenario_id,
            seed=c.seed,
            variant_id=c.variant_id,
            row=refreshed.get(c.row.run_id) if c.row else None,
            error=c.error,
        )
        for c in cells
    ]
    return BatchRunSummary(
        batch_id=batch_id, cells=refreshed_cells, skipped_teams=skipped
    )


def _backfill_composite_scores(
    *,
    batch_id: str,
    store: RunStore,
    epsilon: float = EPSILON,
) -> None:
    """Compute composite_score for every cell, normalised against Mirror.

    Mirror itself receives composite_score = 1.0 by definition (its
    metrics divide into themselves). Cells with no Mirror match in
    the same (scenario, seed, variant) leave composite_score NULL.
    """
    rows = store.list_runs(batch_id=batch_id)
    mirror_index: dict[tuple[str, int, str], RunRow] = {
        (r.scenario_id, r.seed, r.variant_id): r
        for r in rows
        if r.team_id == MIRROR_TEAM_ID
    }
    for row in rows:
        key = (row.scenario_id, row.seed, row.variant_id)
        baseline = mirror_index.get(key)
        if baseline is None:
            continue
        if row.bullwhip_ratio is None or row.total_cost is None:
            continue
        if baseline.bullwhip_ratio is None or baseline.total_cost is None:
            continue
        your = ScoreComponents(
            bullwhip_ratio=row.bullwhip_ratio,
            total_cost=row.total_cost,
            tokens_used=float(row.tokens_used or 0),
        )
        base = ScoreComponents(
            bullwhip_ratio=baseline.bullwhip_ratio,
            total_cost=baseline.total_cost,
            tokens_used=float(baseline.tokens_used or 0),
        )
        breakdown = normalize_components(your, base, epsilon=epsilon)
        store.update_composite_score(row.run_id, breakdown.composite_score)


__all__ = [
    "AgentFactory",
    "CellOutcome",
    "BatchRunSummary",
    "REFERENCE_STARTER_FACTORIES",
    "TeamBuilder",
    "TeamSpec",
    "batch_run",
    "coached_llm_baseline_team_spec",
    "llm_baseline_team_spec",
    "mirror_team_spec",
    "reference_starter_specs",
]
