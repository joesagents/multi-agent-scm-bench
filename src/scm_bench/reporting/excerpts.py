"""Failure-tick selection from RunRecord JSONL.

Two excerpts per scenario:
  (a) the tick with the largest backlog spike (max chain backlog)
  (b) the tick where this team's decision diverged most from the
      Mirror baseline at the same observation

Surfaces a concrete, quotable moment for write-ups instead of a
generic "we needed more time" lament.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scm_bench.engine.state import TIER_NAMES
from scm_bench.trace.store import MIRROR_TEAM_ID, RunStore


@dataclass(frozen=True)
class TickExcerpt:
    scenario_id: str
    seed: int
    tick: int
    label: str
    consumer_demand: int
    decisions: dict[str, int]
    observations_summary: dict[str, dict[str, int]]


def _load_ticks(store: RunStore, jsonl_rel_path: str) -> list[dict]:
    return [
        rec
        for rec in store.read_jsonl(Path(jsonl_rel_path))
        if rec.get("type") == "tick"
    ]


def _chain_backlog(tick: dict) -> int:
    return sum(int(tick["observations"][role]["backlog"]) for role in TIER_NAMES)


def _decision_divergence(team_tick: dict, mirror_tick: dict) -> int:
    return sum(
        abs(
            int(team_tick["decisions"][role]["order_qty"])
            - int(mirror_tick["decisions"][role]["order_qty"])
        )
        for role in TIER_NAMES
    )


def _summarise_observations(tick: dict) -> dict[str, dict[str, int]]:
    return {
        role: {
            "inventory_on_hand": int(obs["inventory_on_hand"]),
            "backlog": int(obs["backlog"]),
            "incoming_order_qty": int(obs["incoming_order_qty"]),
            "incoming_shipment_qty": int(obs["incoming_shipment_qty"]),
        }
        for role, obs in tick["observations"].items()
    }


def excerpts_for_team(
    *,
    store: RunStore,
    batch_id: str,
    team_id: str,
    max_per_scenario: int = 2,
) -> list[TickExcerpt]:
    rows = store.list_runs(batch_id=batch_id, team_id=team_id)
    mirror_rows = store.list_runs(batch_id=batch_id, team_id=MIRROR_TEAM_ID)
    mirror_idx = {
        (r.scenario_id, r.seed, r.variant_id): r for r in mirror_rows
    }
    out: list[TickExcerpt] = []
    seen_scenarios: set[str] = set()
    for r in rows:
        if r.scenario_id in seen_scenarios:
            continue
        seen_scenarios.add(r.scenario_id)
        team_ticks = _load_ticks(store, r.jsonl_path)
        if not team_ticks:
            continue
        # (a) max backlog spike
        spike = max(team_ticks, key=_chain_backlog)
        out.append(
            TickExcerpt(
                scenario_id=r.scenario_id,
                seed=r.seed,
                tick=int(spike["t"]),
                label="largest chain backlog",
                consumer_demand=int(spike["consumer_demand"]),
                decisions={
                    role: int(spike["decisions"][role]["order_qty"])
                    for role in TIER_NAMES
                },
                observations_summary=_summarise_observations(spike),
            )
        )
        if max_per_scenario < 2:
            continue
        # (b) decision divergence vs mirror
        baseline = mirror_idx.get((r.scenario_id, r.seed, r.variant_id))
        if baseline is None:
            continue
        mirror_ticks = _load_ticks(store, baseline.jsonl_path)
        # align by t
        team_by_t = {int(t["t"]): t for t in team_ticks}
        mirror_by_t = {int(t["t"]): t for t in mirror_ticks}
        common = sorted(set(team_by_t) & set(mirror_by_t))
        if not common:
            continue
        worst_t = max(
            common,
            key=lambda t: _decision_divergence(team_by_t[t], mirror_by_t[t]),
        )
        worst = team_by_t[worst_t]
        out.append(
            TickExcerpt(
                scenario_id=r.scenario_id,
                seed=r.seed,
                tick=worst_t,
                label="largest divergence from Mirror baseline",
                consumer_demand=int(worst["consumer_demand"]),
                decisions={
                    role: int(worst["decisions"][role]["order_qty"])
                    for role in TIER_NAMES
                },
                observations_summary=_summarise_observations(worst),
            )
        )
    return out


def render_excerpt(ex: TickExcerpt) -> str:
    """Render a single excerpt as a quotable Markdown code block."""
    body = {
        "scenario_id": ex.scenario_id,
        "seed": ex.seed,
        "tick": ex.tick,
        "consumer_demand": ex.consumer_demand,
        "decisions": ex.decisions,
        "observations": ex.observations_summary,
    }
    return f"**{ex.scenario_id}, seed {ex.seed}, tick {ex.tick} — {ex.label}**\n\n```json\n{json.dumps(body, indent=2)}\n```"


__all__ = ["TickExcerpt", "excerpts_for_team", "render_excerpt"]
