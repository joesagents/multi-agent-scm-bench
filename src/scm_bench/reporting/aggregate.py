"""Population-level aggregate summary.

One Markdown artifact per batch-run (`_aggregate_summary.md`) intended
as a fast-scan triage view across all teams. Tables only — no
editorial narrative.

Sections:
  1. Population leaderboard (one row per team across all scenarios)
  2. Per-tier population distribution (mean / p25 / p75 of token + cost)
  3. Exemplar candidates (best composite, best bullwhip, etc.)
  4. Run-quality audit (which teams failed which cells)
  5. Reproducibility footer
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scm_bench import SDK_VERSION
from scm_bench.engine.state import TIER_NAMES
from scm_bench.reporting.render import (
    fmt_float,
    fmt_int,
    md_html_anchor,
    md_table,
)
from scm_bench.trace.store import MIRROR_TEAM_ID, RunRow, RunStore


@dataclass(frozen=True)
class _TeamAgg:
    team_id: str
    composite_by_scenario: dict[str, float]
    bullwhip_mean: float
    cost_mean: float
    tokens_mean: float
    cells_total: int
    cells_failed: int


def _aggregate_teams(rows: list[RunRow]) -> list[_TeamAgg]:
    by_team: dict[str, list[RunRow]] = defaultdict(list)
    for r in rows:
        by_team[r.team_id].append(r)
    out: list[_TeamAgg] = []
    for team_id, team_rows in by_team.items():
        per_sc: dict[str, list[float]] = defaultdict(list)
        for r in team_rows:
            if r.composite_score is not None:
                per_sc[r.scenario_id].append(r.composite_score)
        composite_by_scenario = {
            sc: statistics.fmean(vals) for sc, vals in per_sc.items()
        }
        bws = [r.bullwhip_ratio for r in team_rows if r.bullwhip_ratio is not None]
        costs = [r.total_cost for r in team_rows if r.total_cost is not None]
        tokens = [r.tokens_used or 0 for r in team_rows]
        out.append(
            _TeamAgg(
                team_id=team_id,
                composite_by_scenario=composite_by_scenario,
                bullwhip_mean=statistics.fmean(bws) if bws else 0.0,
                cost_mean=statistics.fmean(costs) if costs else 0.0,
                tokens_mean=statistics.fmean(tokens),
                cells_total=len(team_rows),
                cells_failed=sum(1 for r in team_rows if r.composite_score is None),
            )
        )
    return out


def _mean_rank(agg: _TeamAgg, ranks: dict[str, dict[str, int]]) -> float | None:
    """Mean rank across scenarios this team participated in."""
    seen = []
    for sc in agg.composite_by_scenario:
        if sc in ranks and agg.team_id in ranks[sc]:
            seen.append(ranks[sc][agg.team_id])
    return statistics.fmean(seen) if seen else None


def _scenario_ranks(aggs: list[_TeamAgg]) -> dict[str, dict[str, int]]:
    """For each scenario, rank teams by composite (lower = better)."""
    by_sc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a in aggs:
        for sc, score in a.composite_by_scenario.items():
            by_sc[sc].append((a.team_id, score))
    out: dict[str, dict[str, int]] = {}
    for sc, pairs in by_sc.items():
        pairs.sort(key=lambda p: p[1])
        out[sc] = {team_id: i + 1 for i, (team_id, _) in enumerate(pairs)}
    return out


def leaderboard_table(aggs: list[_TeamAgg], scenario_ids: list[str]) -> str:
    ranks = _scenario_ranks(aggs)
    aggs_sorted = sorted(
        aggs,
        key=lambda a: (
            _mean_rank(a, ranks) or float("inf"),
            a.team_id,
        ),
    )
    headers = ["team_id", *scenario_ids, "mean rank", "bullwhip", "cost", "tokens"]
    body = []
    for a in aggs_sorted:
        row = [a.team_id]
        for sc in scenario_ids:
            row.append(fmt_float(a.composite_by_scenario.get(sc)))
        mr = _mean_rank(a, ranks)
        row.extend([
            fmt_float(mr, places=2) if mr is not None else "—",
            fmt_float(a.bullwhip_mean),
            fmt_float(a.cost_mean, places=2),
            fmt_int(int(a.tokens_mean)),
        ])
        body.append(row)
    return md_table(headers, body)


def per_tier_distribution_table(
    *, store: RunStore, batch_id: str
) -> str:
    """Population distribution per tier × {avg_inventory, total_cost, tokens}."""
    rows = [
        r for r in store.list_runs(batch_id=batch_id)
        if r.team_id != MIRROR_TEAM_ID
    ]
    inv_by_tier: dict[str, list[float]] = defaultdict(list)
    cost_by_tier: dict[str, list[float]] = defaultdict(list)
    tokens_by_tier: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        for record in store.read_jsonl(r.jsonl_path):
            if record.get("type") == "result":
                for role in TIER_NAMES:
                    tr = record["tier_results"][role]
                    inv_by_tier[role].append(float(tr["avg_inventory"]))
                    cost_by_tier[role].append(float(tr["total_cost"]))
            elif record.get("type") == "tick":
                for role, dec in record["decisions"].items():
                    tokens_by_tier[role].append(int(dec.get("tokens_used", 0)))

    def _stats(xs: list[float]) -> tuple[str, str, str]:
        if not xs:
            return "—", "—", "—"
        xs_sorted = sorted(xs)
        mean = statistics.fmean(xs_sorted)
        n = len(xs_sorted)
        p25 = xs_sorted[max(n * 25 // 100 - 1, 0)]
        p75 = xs_sorted[min(n * 75 // 100, n - 1)]
        return fmt_float(mean, places=2), fmt_float(p25, places=2), fmt_float(p75, places=2)

    body = []
    for role in TIER_NAMES:
        m, p25, p75 = _stats(inv_by_tier[role])
        cm, cp25, cp75 = _stats(cost_by_tier[role])
        tm, tp25, tp75 = _stats([float(t) for t in tokens_by_tier[role]])
        body.append([
            role,
            f"{m} ({p25}/{p75})",
            f"{cm} ({cp25}/{cp75})",
            f"{tm} ({tp25}/{tp75})",
        ])
    return md_table(
        ["Tier", "avg_inventory mean (p25/p75)",
         "total_cost mean (p25/p75)", "tokens_used mean (p25/p75)"],
        body,
    )


def exemplar_candidates_table(aggs: list[_TeamAgg]) -> str:
    student_aggs = [
        a for a in aggs
        if not a.team_id.startswith("__") and a.composite_by_scenario
    ]
    if not student_aggs:
        return "_no submitted teams in this batch-run._"

    def best_composite(a: _TeamAgg) -> float:
        return statistics.fmean(a.composite_by_scenario.values())

    def stability_ratio(a: _TeamAgg) -> float:
        s1_1 = a.composite_by_scenario.get("s1.1")
        s2_3 = a.composite_by_scenario.get("s2.3")
        if s1_1 is None or s2_3 is None or s1_1 == 0:
            return float("inf")
        return s2_3 / s1_1

    rows = [
        ["best mean composite",
         min(student_aggs, key=best_composite).team_id,
         "lowest mean composite across scenarios"],
        ["best bullwhip dampening",
         min(student_aggs, key=lambda a: a.bullwhip_mean).team_id,
         "lowest mean bullwhip_ratio"],
        ["lowest token spend",
         min(student_aggs, key=lambda a: a.tokens_mean).team_id,
         "fewest tokens used per run"],
        ["least stable (S2.3 vs S1.1)",
         max(student_aggs, key=stability_ratio).team_id,
         "largest composite ratio s2.3/s1.1"],
    ]
    return md_table(["Pick", "Team", "Why"], rows)


def run_quality_table(*, store: RunStore, batch_id: str) -> str:
    rows = store.list_runs(batch_id=batch_id)
    by_team: dict[str, list[RunRow]] = defaultdict(list)
    for r in rows:
        by_team[r.team_id].append(r)
    body = []
    for team_id in sorted(by_team):
        team_rows = by_team[team_id]
        total = len(team_rows)
        with_score = sum(1 for r in team_rows if r.composite_score is not None)
        body.append([
            team_id,
            str(total),
            str(with_score),
            "OK" if with_score == total else "PARTIAL",
        ])
    return md_table(["Team", "Cells", "Cells with composite", "Status"], body)


def reproducibility_footer(
    *, batch_id: str, store: RunStore, git_sha: str | None
) -> str:
    rows = store.list_runs(batch_id=batch_id)
    scenarios = sorted({r.scenario_id for r in rows})
    seeds = sorted({r.seed for r in rows})
    teams = sorted({r.team_id for r in rows})
    body = [
        ["batch_id", batch_id],
        ["scenarios", ", ".join(scenarios)],
        ["seeds", ", ".join(str(s) for s in seeds)],
        ["teams", str(len(teams))],
        ["sdk_version", SDK_VERSION],
        ["git_sha", git_sha or "—"],
        ["generated_at", datetime.now(UTC).isoformat(timespec="seconds")],
    ]
    return md_table(["Field", "Value"], body)


def write_aggregate_summary(
    *,
    store: RunStore,
    batch_id: str,
    out_root: Path,
    git_sha: str | None = None,
) -> Path:
    rows = store.list_runs(batch_id=batch_id)
    if not rows:
        raise ValueError(
            f"no runs found for batch_id={batch_id!r} — "
            "run `scm_bench batch-run` first."
        )
    aggs = _aggregate_teams(rows)
    scenario_ids = sorted({r.scenario_id for r in rows})

    sections = [
        ("Population leaderboard",
         "writeup/_aggregate/leaderboard",
         leaderboard_table(aggs, scenario_ids)),
        ("Per-tier population distribution",
         "writeup/_aggregate/per_tier",
         per_tier_distribution_table(store=store, batch_id=batch_id)),
        ("Exemplar candidates",
         "writeup/_aggregate/exemplars",
         exemplar_candidates_table(aggs)),
        ("Run-quality audit",
         "writeup/_aggregate/audit",
         run_quality_table(store=store, batch_id=batch_id)),
        ("Reproducibility footer",
         "writeup/_aggregate/footer",
         reproducibility_footer(
             batch_id=batch_id, store=store, git_sha=git_sha
         )),
    ]

    parts: list[str] = [
        f"# Aggregate summary — batch-run `{batch_id}`",
        "",
        f"_{len(aggs)} teams, {len(scenario_ids)} scenarios, "
        f"{len({r.seed for r in rows})} seeds._",
        "",
    ]
    for title, maps_to, body in sections:
        parts.append(f"## {title}\n\n{md_html_anchor(maps_to=maps_to)}\n\n{body}")
        parts.append("")

    out_path = out_root / batch_id / "_aggregate_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts).rstrip() + "\n")
    return out_path


__all__ = [
    "leaderboard_table",
    "per_tier_distribution_table",
    "exemplar_candidates_table",
    "run_quality_table",
    "reproducibility_footer",
    "write_aggregate_summary",
]
