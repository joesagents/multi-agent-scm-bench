"""End-to-end batch-run smoke against the four reference starters.

Runs the full {Mirror, MovingAverage, BaseStock, CommunicatingForecast}
× {S1.1, S2.3} × seeds 0-1 = 16 cells through the batch orchestrator
and the JSONL/SQLite trace store. Mirror's composite_score must be
exactly 1.0 on every cell (it is the baseline by definition).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scm_bench.runner.batch import (
    REFERENCE_STARTER_FACTORIES,
    TeamSpec,
    batch_run,
)
from scm_bench.trace.store import MIRROR_TEAM_ID, RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path)


def test_starter_batch_run_smoke(store: RunStore) -> None:
    starter_teams = [
        TeamSpec(team_id=team_id, factory=factory)
        for team_id, factory in REFERENCE_STARTER_FACTORIES.items()
    ]
    summary = batch_run(
        batch_id="smoke-2026-04",
        teams=starter_teams,
        scenario_ids=["s1.1", "s2.3"],
        seeds=[0, 1],
        store=store,
        include_mirror=True,
    )

    # Mirror + 3 reference starters = 4 teams; × 2 scenarios × 2 seeds = 16 cells.
    expected_cells = (1 + len(REFERENCE_STARTER_FACTORIES)) * 2 * 2
    assert len(summary.cells) == expected_cells
    assert all(c.ok for c in summary.cells)
    assert summary.skipped_teams == []


def test_mirror_baseline_composite_is_one(store: RunStore) -> None:
    starter_teams = [
        TeamSpec(team_id=team_id, factory=factory)
        for team_id, factory in REFERENCE_STARTER_FACTORIES.items()
    ]
    batch_run(
        batch_id="smoke-mirror",
        teams=starter_teams,
        scenario_ids=["s1.1"],
        seeds=[0],
        store=store,
        include_mirror=True,
    )
    rows = store.list_runs(batch_id="smoke-mirror", team_id=MIRROR_TEAM_ID)
    assert rows
    for r in rows:
        assert r.composite_score == pytest.approx(1.0)


def test_batch_run_persists_jsonl_and_db(store: RunStore) -> None:
    summary = batch_run(
        batch_id="smoke-persist",
        teams=[],
        scenario_ids=["s1.1"],
        seeds=[0],
        store=store,
        include_mirror=True,
    )
    assert len(summary.cells) == 1
    cell = summary.cells[0]
    assert cell.ok
    assert cell.row is not None
    jsonl_path = store.root / cell.row.jsonl_path
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    # header + N tick records + result
    assert len(lines) >= 3
    assert '"type": "header"' in lines[0]
    assert '"type": "result"' in lines[-1]
