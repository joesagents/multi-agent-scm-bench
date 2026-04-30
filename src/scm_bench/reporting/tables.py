"""Pure-data builders for the per-team rundown's seven mandatory tables.

Each builder consumes (RunStore rows + JSONL traces) and returns a
Markdown string. The per-team assembler concatenates them with the
required ``<!-- maps to: writeup/... -->`` anchors.

Sections, in writeup-template order:
  1. Headline numbers      (opening line of <sc>_results.txt)
  2. Strategy fingerprint  (<sc>_design.txt — observed order pattern)
  3. Chain-level results   (<sc>_results.txt mandatory Table 1)
  4. Per-tier results      (<sc>_results.txt mandatory Table 2)
  5. Stability check       (<sc>_results.txt mandatory Table 3)
  6. Sensitivity scaffold  (<sc>_results.txt mandatory Table 4)
  7. Failure excerpts      (<sc>_results.txt reflection paragraph)
  8. Baselines comparison  (Section A coordination + Section B alternatives)
  9. Reproducibility footer
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from scm_bench import SDK_VERSION
from scm_bench.engine.state import TIER_NAMES
from scm_bench.metrics.composite import (
    EPSILON,
    ScoreComponents,
    normalize_components,
)
from scm_bench.metrics.stability import direction_held, quartile
from scm_bench.reporting.render import (
    fmt_float,
    fmt_int,
    md_html_anchor,
    md_table,
)
from scm_bench.trace.store import MIRROR_TEAM_ID, RunRow, RunStore


@dataclass(frozen=True)
class TeamView:
    """Pre-aggregated rows for one team within a batch-run."""

    team_id: str
    rows: list[RunRow]
    mirror_index: dict[tuple[str, int, str], RunRow]
    population_composite_by_scenario: dict[str, list[float]]

    def rows_for(self, scenario_id: str) -> list[RunRow]:
        return [r for r in self.rows if r.scenario_id == scenario_id]

    def scenario_ids(self) -> list[str]:
        return sorted({r.scenario_id for r in self.rows})


def build_team_view(
    *,
    store: RunStore,
    batch_id: str,
    team_id: str,
) -> TeamView:
    all_rows = store.list_runs(batch_id=batch_id)
    team_rows = [r for r in all_rows if r.team_id == team_id]
    mirror_rows = [r for r in all_rows if r.team_id == MIRROR_TEAM_ID]
    mirror_idx = {
        (r.scenario_id, r.seed, r.variant_id): r for r in mirror_rows
    }
    population: dict[str, list[float]] = defaultdict(list)
    for r in all_rows:
        if r.team_id == MIRROR_TEAM_ID:
            continue
        if r.composite_score is not None:
            population[r.scenario_id].append(r.composite_score)
    return TeamView(
        team_id=team_id,
        rows=team_rows,
        mirror_index=mirror_idx,
        population_composite_by_scenario=dict(population),
    )


# ----------------------------------------------------------------------
# Table 1 — headline numbers
# ----------------------------------------------------------------------


def headline_table(view: TeamView) -> str:
    """Composite per scenario + population rank + pass/fail per level."""
    lines = []
    for scenario_id in view.scenario_ids():
        rows = view.rows_for(scenario_id)
        composites = [r.composite_score for r in rows if r.composite_score is not None]
        mean = statistics.fmean(composites) if composites else None
        population = view.population_composite_by_scenario.get(scenario_id, [])
        q = quartile(mean, population) if mean is not None else 0
        passed = "PASS" if mean is not None and mean <= 1.0 else "—"
        lines.append([
            scenario_id,
            fmt_float(mean, places=3),
            f"Q{q}" if q else "—",
            passed,
        ])
    return md_table(
        ["Scenario", "Composite (mean over seeds)", "Population quartile", "Beat baseline?"],
        lines,
    )


# ----------------------------------------------------------------------
# Table 2 — strategy fingerprint per tier
# ----------------------------------------------------------------------


def strategy_fingerprint_table(
    *, view: TeamView, store: RunStore
) -> str:
    """Per-tier observed order stats across the team's runs."""
    by_tier: dict[str, list[int]] = {role: [] for role in TIER_NAMES}
    for r in view.rows:
        for record in store.read_jsonl(r.jsonl_path):
            if record.get("type") != "tick":
                continue
            for role in TIER_NAMES:
                dec = record["decisions"].get(role)
                if dec is not None:
                    by_tier[role].append(int(dec["order_qty"]))
    rows = []
    for role in TIER_NAMES:
        orders = by_tier[role]
        mean = statistics.fmean(orders) if orders else 0.0
        stdev = statistics.pstdev(orders) if len(orders) > 1 else 0.0
        cv = (stdev / mean) if mean > 0 else 0.0
        rows.append([
            role,
            fmt_float(mean, places=2),
            fmt_float(stdev, places=2),
            fmt_float(cv, places=3),
        ])
    return md_table(
        ["Tier", "Mean order qty", "Std order qty", "Order CV"],
        rows,
    )


# ----------------------------------------------------------------------
# Table 3 — chain-level results (one block per scenario)
# ----------------------------------------------------------------------


def _aggregate_team_components(rows: list[RunRow]) -> ScoreComponents | None:
    if not rows:
        return None
    bullwhip = statistics.fmean(
        r.bullwhip_ratio for r in rows if r.bullwhip_ratio is not None
    )
    cost = statistics.fmean(
        r.total_cost for r in rows if r.total_cost is not None
    )
    tokens = statistics.fmean(
        (r.tokens_used or 0) for r in rows
    )
    return ScoreComponents(
        bullwhip_ratio=bullwhip, total_cost=cost, tokens_used=tokens
    )


def chain_results_table(view: TeamView) -> str:
    blocks: list[str] = []
    for scenario_id in view.scenario_ids():
        team_rows = view.rows_for(scenario_id)
        mirror_rows = [
            view.mirror_index[(scenario_id, r.seed, r.variant_id)]
            for r in team_rows
            if (scenario_id, r.seed, r.variant_id) in view.mirror_index
        ]
        team_comp = _aggregate_team_components(team_rows)
        mirror_comp = _aggregate_team_components(mirror_rows)
        chain_fill = statistics.fmean(
            r.chain_fill_rate for r in team_rows if r.chain_fill_rate is not None
        )
        if team_comp is None or mirror_comp is None:
            blocks.append(f"### {scenario_id}\n\n_no comparable data._")
            continue
        breakdown = normalize_components(team_comp, mirror_comp, epsilon=EPSILON)
        rows = [
            ["composite_score",
             fmt_float(breakdown.composite_score),
             "1.000",
             fmt_float(breakdown.composite_score)],
            ["bullwhip_ratio",
             fmt_float(team_comp.bullwhip_ratio),
             fmt_float(mirror_comp.bullwhip_ratio),
             fmt_float(breakdown.normalized_bullwhip)],
            ["total_cost",
             fmt_float(team_comp.total_cost, places=2),
             fmt_float(mirror_comp.total_cost, places=2),
             fmt_float(breakdown.normalized_cost)],
            ["tokens_used",
             fmt_int(int(team_comp.tokens_used)),
             fmt_int(int(mirror_comp.tokens_used)),
             fmt_float(breakdown.normalized_tokens)],
            ["chain_fill_rate",
             fmt_float(chain_fill),
             "—",
             "—"],
        ]
        blocks.append(
            f"### {scenario_id}\n\n"
            + md_table(
                ["Metric", "Your Agent", "Echo (Mirror)", "Normalized"],
                rows,
            )
        )
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------
# Table 4 — per-tier results
# ----------------------------------------------------------------------


def per_tier_results_table(*, view: TeamView, store: RunStore) -> str:
    blocks: list[str] = []
    for scenario_id in view.scenario_ids():
        rows = view.rows_for(scenario_id)
        per_tier_acc: dict[str, dict[str, list[float]]] = {
            role: defaultdict(list) for role in TIER_NAMES
        }
        for r in rows:
            for record in store.read_jsonl(r.jsonl_path):
                if record.get("type") != "result":
                    continue
                for role in TIER_NAMES:
                    tr = record["tier_results"][role]
                    per_tier_acc[role]["holding"].append(tr["total_holding_cost"])
                    per_tier_acc[role]["backlog"].append(tr["total_backlog_cost"])
                    per_tier_acc[role]["total"].append(tr["total_cost"])
                    per_tier_acc[role]["fill"].append(tr["fill_rate"])
                    per_tier_acc[role]["inv"].append(tr["avg_inventory"])
                    per_tier_acc[role]["cv"].append(tr["order_cv"])
        # Tokens per tier across cells
        token_acc: dict[str, list[int]] = {role: [] for role in TIER_NAMES}
        for r in rows:
            run_tokens: dict[str, int] = {role: 0 for role in TIER_NAMES}
            for record in store.read_jsonl(r.jsonl_path):
                if record.get("type") != "tick":
                    continue
                for role, dec in record["decisions"].items():
                    run_tokens[role] += int(dec.get("tokens_used", 0))
            for role in TIER_NAMES:
                token_acc[role].append(run_tokens[role])
        body_rows = []
        for role in TIER_NAMES:
            acc = per_tier_acc[role]
            body_rows.append([
                role,
                fmt_float(statistics.fmean(acc["holding"]) if acc["holding"] else 0.0, places=2),
                fmt_float(statistics.fmean(acc["backlog"]) if acc["backlog"] else 0.0, places=2),
                fmt_float(statistics.fmean(acc["total"]) if acc["total"] else 0.0, places=2),
                fmt_float(statistics.fmean(acc["fill"]) if acc["fill"] else 0.0),
                fmt_float(statistics.fmean(acc["inv"]) if acc["inv"] else 0.0, places=2),
                fmt_float(statistics.fmean(acc["cv"]) if acc["cv"] else 0.0),
                fmt_int(int(statistics.fmean(token_acc[role])) if token_acc[role] else 0),
                "—",  # messages_sent (Phase-2 messaging bus)
                "—",  # messages_received
            ])
        blocks.append(
            f"### {scenario_id}\n\n"
            + md_table(
                [
                    "Tier", "holding_cost", "backlog_cost", "total_cost",
                    "fill_rate", "avg_inventory", "order_cv", "tokens_used",
                    "messages_sent", "messages_received",
                ],
                body_rows,
            )
        )
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------
# Table 5 — stability check (S1.1 vs S2.3)
# ----------------------------------------------------------------------


def stability_table(view: TeamView) -> str:
    s1_1 = view.population_composite_by_scenario.get("s1.1", [])
    s2_3 = view.population_composite_by_scenario.get("s2.3", [])
    s1_1_rows = view.rows_for("s1.1")
    s2_3_rows = view.rows_for("s2.3")
    if not s1_1_rows or not s2_3_rows:
        return "_stability check requires runs on both s1.1 and s2.3._"
    your_s1_1 = statistics.fmean(
        r.composite_score for r in s1_1_rows if r.composite_score is not None
    )
    your_s2_3 = statistics.fmean(
        r.composite_score for r in s2_3_rows if r.composite_score is not None
    )
    cost_s1_1 = statistics.fmean(
        r.total_cost for r in s1_1_rows if r.total_cost is not None
    )
    cost_s2_3 = statistics.fmean(
        r.total_cost for r in s2_3_rows if r.total_cost is not None
    )
    bw_s1_1 = statistics.fmean(
        r.bullwhip_ratio for r in s1_1_rows if r.bullwhip_ratio is not None
    )
    bw_s2_3 = statistics.fmean(
        r.bullwhip_ratio for r in s2_3_rows if r.bullwhip_ratio is not None
    )
    fill_s1_1 = statistics.fmean(
        r.chain_fill_rate for r in s1_1_rows if r.chain_fill_rate is not None
    )
    fill_s2_3 = statistics.fmean(
        r.chain_fill_rate for r in s2_3_rows if r.chain_fill_rate is not None
    )
    held = direction_held(
        team_score_stable=your_s1_1,
        team_score_disrupted=your_s2_3,
        population_scores_stable=s1_1,
        population_scores_disrupted=s2_3,
    )
    rows = [
        ["composite_score", fmt_float(your_s1_1), fmt_float(your_s2_3)],
        ["bullwhip_ratio", fmt_float(bw_s1_1), fmt_float(bw_s2_3)],
        ["total_cost", fmt_float(cost_s1_1, places=2), fmt_float(cost_s2_3, places=2)],
        ["chain_fill_rate", fmt_float(fill_s1_1), fmt_float(fill_s2_3)],
    ]
    held_str = "**yes**" if held else "**no**"
    suffix = (
        f"\n\nPopulation quartile direction held? {held_str} "
        "(quartile rank on s1.1 vs s2.3 — same → robust strategy.)"
    )
    return md_table(
        ["Metric", "S1.1 (stable)", "S2.3 (disruption)"], rows
    ) + suffix


# ----------------------------------------------------------------------
# Table 6 — sensitivity scaffold (auto-fills with variants if present)
# ----------------------------------------------------------------------


def sensitivity_table(view: TeamView) -> str:
    variants = sorted({r.variant_id for r in view.rows if r.variant_id})
    if len(variants) < 1:
        return (
            md_table(
                ["Component changed", "Before", "After", "Explanation"],
                [["", "", "", ""]],
            )
            + "\n\n_Run `scm_bench test-bundle <bundle> --variant <name>` "
            "to populate this table automatically with a before/after pair._"
        )
    rows = []
    base_rows = [r for r in view.rows if not r.variant_id]
    base_score = statistics.fmean(
        r.composite_score for r in base_rows if r.composite_score is not None
    )
    for v in variants:
        v_rows = [r for r in view.rows if r.variant_id == v]
        v_score = statistics.fmean(
            r.composite_score for r in v_rows if r.composite_score is not None
        )
        rows.append([v, fmt_float(base_score), fmt_float(v_score), ""])
    return md_table(
        ["Component changed", "Before", "After", "Explanation"], rows
    )


# ----------------------------------------------------------------------
# Table 7 — failure excerpts (handled in excerpts.py and inlined here)
# ----------------------------------------------------------------------
# (assembled by per_team.py)


# ----------------------------------------------------------------------
# Table 8 — baselines comparison (composite per scenario, baselines as rows)
# ----------------------------------------------------------------------


def baselines_comparison_table(
    *, store: RunStore, batch_id: str, team_id: str
) -> str:
    all_rows = store.list_runs(batch_id=batch_id)
    scenarios = sorted({r.scenario_id for r in all_rows})
    ordered = [team_id, MIRROR_TEAM_ID] + sorted(
        {
            r.team_id
            for r in all_rows
            if r.team_id.startswith("__starter_") or r.team_id.startswith("__llm_")
        }
    )
    teams_to_show: list[str] = []
    for t in ordered:
        if t not in teams_to_show:
            teams_to_show.append(t)
    body: list[list[str]] = []
    for t in teams_to_show:
        row = [t]
        for sc in scenarios:
            cells = [
                r for r in all_rows
                if r.team_id == t and r.scenario_id == sc and r.composite_score is not None
            ]
            row.append(
                fmt_float(statistics.fmean(r.composite_score for r in cells))
                if cells
                else "—"
            )
        body.append(row)
    return md_table(["Team"] + scenarios, body)


# ----------------------------------------------------------------------
# Table 9 — reproducibility footer
# ----------------------------------------------------------------------


def reproducibility_table(
    *, batch_id: str, view: TeamView, git_sha: str | None = None
) -> str:
    seeds = sorted({r.seed for r in view.rows})
    scenarios = sorted({r.scenario_id for r in view.rows})
    rows = [
        ["batch_id", batch_id],
        ["scenarios", ", ".join(scenarios)],
        ["seeds", ", ".join(str(s) for s in seeds)],
        ["sdk_version", SDK_VERSION],
        ["git_sha", git_sha or "—"],
        ["generated_at", datetime.now(UTC).isoformat(timespec="seconds")],
    ]
    return md_table(["Field", "Value"], rows)


__all__ = [
    "TeamView",
    "build_team_view",
    "headline_table",
    "strategy_fingerprint_table",
    "chain_results_table",
    "per_tier_results_table",
    "stability_table",
    "sensitivity_table",
    "baselines_comparison_table",
    "reproducibility_table",
    "md_html_anchor",
]


def _ensure_iter(x: Iterable) -> list:  # used for typing only — kept for compat
    return list(x)
