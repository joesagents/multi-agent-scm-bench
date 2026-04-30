"""Per-team rundown + aggregate summary smoke tests.

Drives a small batch-run end-to-end (mirror echo + the two non-LLM
starters) on s1.1 + s2.3, then asserts that:
- one rundown.md per team carries every mandatory writeup section
- per-section table snippets land in tables/
- the aggregate summary lists every team and every scenario
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scm_bench.reporting.aggregate import write_aggregate_summary
from scm_bench.reporting.per_team import (
    team_ids_in_batch,
    write_team_rundown,
)
from scm_bench.runner.batch import (
    TeamSpec,
    batch_run as batch_orchestrator,
    mirror_team_spec,
)
from scm_bench.starters.base_stock import BaseStockAgent
from scm_bench.starters.moving_average import MovingAverageAgent
from scm_bench.trace.store import MIRROR_TEAM_ID, RunStore


BATCH_ID = "test-p2b-smoke"
SCENARIO_IDS = ["s1.1", "s2.3"]
SEEDS = [0, 1]


@pytest.fixture(scope="module")
def populated_store(tmp_path_factory: pytest.TempPathFactory) -> RunStore:
    store_root = tmp_path_factory.mktemp("runstore")
    store = RunStore(store_root)
    teams = [
        mirror_team_spec(),
        TeamSpec(team_id="__starter_moving_average__", factory=MovingAverageAgent),
        TeamSpec(team_id="__starter_base_stock__", factory=BaseStockAgent),
    ]
    batch_orchestrator(
        batch_id=BATCH_ID,
        teams=teams,
        scenario_ids=SCENARIO_IDS,
        seeds=SEEDS,
        store=store,
        include_mirror=True,
    )
    return store


def test_per_team_rundown_contains_all_mandatory_sections(
    populated_store: RunStore, tmp_path: Path
) -> None:
    rundown = write_team_rundown(
        store=populated_store,
        batch_id=BATCH_ID,
        team_id="__starter_moving_average__",
        out_root=tmp_path,
        git_sha="deadbeef",
    )
    assert rundown.exists()
    text = rundown.read_text()
    for needle in (
        "Headline numbers",
        "Per-tier strategy fingerprint",
        "Chain-level results",
        "Per-tier results",
        "Stability check",
        "Sensitivity test",
        "Failure tick excerpts",
        "Comparison vs baselines",
        "Reproducibility footer",
    ):
        assert needle in text, f"missing section: {needle!r}\n---\n{text}"

    # writeup-anchor comments are present and at least one points to _results.txt
    assert "<!-- maps to:" in text
    assert "writeup/<sc>_results.txt" in text

    # Composite_score for at least one scenario was filled (not "—") — the
    # table builder rounds to 3 decimals so any digit in [0,9] works.
    assert " 0." in text or " 1." in text or " 2." in text

    # Reproducibility footer carries the supplied git_sha
    assert "deadbeef" in text


def test_per_team_rundown_writes_one_table_snippet_per_section(
    populated_store: RunStore, tmp_path: Path
) -> None:
    write_team_rundown(
        store=populated_store,
        batch_id=BATCH_ID,
        team_id="__starter_base_stock__",
        out_root=tmp_path,
    )
    tables_dir = tmp_path / BATCH_ID / "__starter_base_stock__" / "tables"
    snippets = sorted(p.name for p in tables_dir.glob("*.md"))
    assert snippets == [
        "01__headline.md",
        "02__strategy_fingerprint.md",
        "03__chain_results.md",
        "04__per_tier_results.md",
        "05__stability.md",
        "06__sensitivity.md",
        "07__failure_excerpts.md",
        "08__baselines.md",
        "09__reproducibility.md",
    ]
    headline = (tables_dir / "01__headline.md").read_text()
    assert "Headline numbers" in headline


def test_team_ids_in_batch_excludes_baselines(
    populated_store: RunStore,
) -> None:
    teams = team_ids_in_batch(
        store=populated_store,
        batch_id=BATCH_ID,
        exclude={MIRROR_TEAM_ID},
    )
    assert MIRROR_TEAM_ID not in teams
    assert "__starter_moving_average__" in teams
    assert "__starter_base_stock__" in teams


def test_aggregate_summary_lists_every_team_and_scenario(
    populated_store: RunStore, tmp_path: Path
) -> None:
    summary_path = write_aggregate_summary(
        store=populated_store,
        batch_id=BATCH_ID,
        out_root=tmp_path,
        git_sha="cafef00d",
    )
    assert summary_path.exists()
    text = summary_path.read_text()

    # Every team should appear in the leaderboard
    assert MIRROR_TEAM_ID in text
    assert "__starter_moving_average__" in text
    assert "__starter_base_stock__" in text

    # Every scenario appears as a leaderboard column header
    for sc in SCENARIO_IDS:
        assert sc in text

    # Section headings are present
    for needle in (
        "Population leaderboard",
        "Per-tier population distribution",
        "Exemplar candidates",
        "Run-quality audit",
        "Reproducibility footer",
    ):
        assert needle in text, f"missing aggregate section: {needle!r}"

    # Reproducibility footer carries the supplied git_sha
    assert "cafef00d" in text


def test_aggregate_summary_errors_on_unknown_batch_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match="no runs found"):
        write_aggregate_summary(
            store=store, batch_id="does-not-exist", out_root=tmp_path
        )
