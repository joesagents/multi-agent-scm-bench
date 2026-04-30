"""RunStore — write/list/import round-trips, composite update, schema."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scm_bench import SDK_VERSION
from scm_bench.runner.harness import RunRecord, run_one
from scm_bench.scenarios import builtin as scenarios_builtin
from scm_bench.starters.mirror import MirrorAgent
from scm_bench.trace.store import (
    MIRROR_TEAM_ID,
    RunRow,
    RunStore,
)


def _record(run_id: str = "r-1", seed: int = 0) -> RunRecord:
    """Run a 5-tick mirror smoke run to get a realistic RunRecord."""
    agents = {
        role: MirrorAgent()
        for role in ("retailer", "wholesaler", "distributor", "factory")
    }
    return run_one(
        agents=agents,
        scenario=scenarios_builtin.INTRO_STEP_DEMAND,
        seed=seed,
        run_id=run_id,
        max_ticks=5,
    )


def test_runstore_creates_db_on_init(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    assert store.db_path.exists()
    with sqlite3.connect(store.db_path) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
    assert "runs" in names


def test_write_run_round_trips_through_list_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    rec = _record(run_id="round-trip")
    row = store.write_run(
        record=rec,
        batch_id="cls-1",
        team_id="team-A",
        sdk_version=SDK_VERSION,
    )
    assert row.run_id == "round-trip"
    assert row.composite_score is None  # not yet backfilled

    rows = store.list_runs(batch_id="cls-1")
    assert len(rows) == 1
    fetched = rows[0]
    assert fetched.team_id == "team-A"
    assert fetched.scenario_id == rec.scenario_id
    assert fetched.seed == 0
    assert fetched.total_cost == row.total_cost


def test_write_run_emits_jsonl_with_header_ticks_result(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    rec = _record()
    row = store.write_run(
        record=rec,
        batch_id="cls-1",
        team_id="team-A",
        sdk_version=SDK_VERSION,
    )
    jsonl = store.root / row.jsonl_path
    payload = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    assert payload[0]["type"] == "header"
    assert payload[0]["sdk_version"] == SDK_VERSION
    assert payload[-1]["type"] == "result"
    tick_records = [r for r in payload if r["type"] == "tick"]
    assert len(tick_records) == len(rec.ticks)


def test_jsonl_path_includes_variant_id_when_supplied(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    base = store.jsonl_path_for(
        batch_id="cls", team_id="t", scenario_id="s1.1", seed=0, variant_id=""
    )
    variant = store.jsonl_path_for(
        batch_id="cls", team_id="t", scenario_id="s1.1", seed=0, variant_id="v1"
    )
    assert "__v1" in variant.name
    assert "__v1" not in base.name


def test_list_runs_filters_by_team(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    for team in ("team-A", "team-B", MIRROR_TEAM_ID):
        store.write_run(
            record=_record(run_id=f"r-{team}"),
            batch_id="cls",
            team_id=team,
            sdk_version=SDK_VERSION,
        )
    only_a = store.list_runs(batch_id="cls", team_id="team-A")
    assert len(only_a) == 1 and only_a[0].team_id == "team-A"
    mirror = store.list_runs(batch_id="cls", team_id=MIRROR_TEAM_ID)
    assert len(mirror) == 1 and mirror[0].team_id == MIRROR_TEAM_ID


def test_list_runs_filters_by_scenario(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    store.write_run(
        record=_record(run_id="x"),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    matched = store.list_runs(batch_id="cls", scenario_id="intro_step_demand")
    assert len(matched) == 1
    miss = store.list_runs(batch_id="cls", scenario_id="s99.9")
    assert miss == []


def test_update_composite_score_persists(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    row = store.write_run(
        record=_record(run_id="comp-1"),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    assert row.composite_score is None
    store.update_composite_score("comp-1", 0.873)
    refreshed = store.list_runs(batch_id="cls")
    assert refreshed[0].composite_score == pytest.approx(0.873)


def test_list_batches_returns_distinct(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    for cls in ("cls-1", "cls-2", "cls-1"):
        store.write_run(
            record=_record(run_id=f"r-{cls}-{id(object())}"),
            batch_id=cls,
            team_id="t",
            sdk_version=SDK_VERSION,
        )
    assert store.list_batches() == ["cls-1", "cls-2"]


def test_unique_constraint_replaces_existing_cell(tmp_path: Path) -> None:
    """Same (batch_run, team, scenario, seed, variant) re-writes the row, not a duplicate."""
    store = RunStore(tmp_path / "store")
    store.write_run(
        record=_record(run_id="first"),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    store.write_run(
        record=_record(run_id="second"),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    rows = store.list_runs(batch_id="cls")
    assert len(rows) == 1
    assert rows[0].run_id == "second"


def test_read_jsonl_reproduces_written_lines(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    row = store.write_run(
        record=_record(),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    records = list(store.read_jsonl(row.jsonl_path))
    assert records[0]["type"] == "header"
    assert records[-1]["type"] == "result"


def test_import_jsonl_round_trips_through_a_second_store(tmp_path: Path) -> None:
    src = RunStore(tmp_path / "src")
    written = src.write_run(
        record=_record(run_id="ingest-me"),
        batch_id="cls-orig",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    jsonl_src = src.root / written.jsonl_path

    dest = RunStore(tmp_path / "dest")
    imported = dest.import_jsonl(jsonl_src)

    assert imported.run_id == "ingest-me"
    assert imported.team_id == "t"
    assert imported.scenario_id == written.scenario_id
    assert imported.total_cost == pytest.approx(written.total_cost)
    # JSONL file should now exist in the dest store at canonical path
    assert (dest.root / imported.jsonl_path).exists()


def test_import_jsonl_batch_id_override_rewrites_header(tmp_path: Path) -> None:
    src = RunStore(tmp_path / "src")
    written = src.write_run(
        record=_record(run_id="ovr"),
        batch_id="orig-class",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    jsonl_src = src.root / written.jsonl_path

    dest = RunStore(tmp_path / "dest")
    imported = dest.import_jsonl(jsonl_src, batch_id_override="renamed-class")

    assert imported.batch_id == "renamed-class"
    # Header in the destination JSONL must reflect the override
    header = next(iter(dest.read_jsonl(imported.jsonl_path)))
    assert header["batch_id"] == "renamed-class"


def test_import_jsonl_rejects_missing_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"type": "tick", "t": 0}) + "\n")
    dest = RunStore(tmp_path / "dest")
    with pytest.raises(ValueError, match="header"):
        dest.import_jsonl(bad)


def test_runrow_dataclass_round_trips_via_dict(tmp_path: Path) -> None:
    """SELECT * → RunRow construction must accept exactly the schema columns."""
    store = RunStore(tmp_path / "store")
    store.write_run(
        record=_record(),
        batch_id="cls",
        team_id="t",
        sdk_version=SDK_VERSION,
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        raw = conn.execute("SELECT * FROM runs LIMIT 1").fetchone()
    row = RunRow(**dict(raw))
    assert row.team_id == "t"


def test_mirror_team_id_constant_is_unique_sentinel() -> None:
    """The mirror baseline team_id must not collide with a plausible real team_id."""
    assert MIRROR_TEAM_ID.startswith("__") and MIRROR_TEAM_ID.endswith("__")
    assert "_baseline_" in MIRROR_TEAM_ID
