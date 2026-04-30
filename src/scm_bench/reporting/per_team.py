"""Per-team rundown assembler.

Concatenates the table builders from `tables.py` and the failure-tick
excerpts from `excerpts.py` into a single `rundown.md`, then writes one
`tables/<n>__<slug>.md` per section so agents can `cat` individual
tables straight into `<sc>_results.txt`.

Layout produced::

    reports/<batch_id>/<team_id>/
        rundown.md
        tables/
            01__headline.md
            02__strategy_fingerprint.md
            03__chain_results.md
            04__per_tier_results.md
            05__stability.md
            06__sensitivity.md
            07__failure_excerpts.md
            08__baselines.md
            09__reproducibility.md

Each section in `rundown.md` carries a `<!-- maps to: writeup/... -->`
HTML anchor so a graders can jump from a writeup section back to the
corresponding evidence block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scm_bench.reporting.excerpts import (
    excerpts_for_team,
    render_excerpt,
)
from scm_bench.reporting.render import md_html_anchor, md_section
from scm_bench.reporting.tables import (
    baselines_comparison_table,
    build_team_view,
    chain_results_table,
    headline_table,
    per_tier_results_table,
    reproducibility_table,
    sensitivity_table,
    stability_table,
    strategy_fingerprint_table,
)
from scm_bench.trace.store import RunStore


@dataclass(frozen=True)
class _Section:
    slug: str
    title: str
    maps_to: str
    body: str


def _build_sections(
    *,
    store: RunStore,
    batch_id: str,
    team_id: str,
    git_sha: str | None,
) -> list[_Section]:
    view = build_team_view(
        store=store, batch_id=batch_id, team_id=team_id
    )
    excerpts = excerpts_for_team(
        store=store, batch_id=batch_id, team_id=team_id
    )
    excerpts_body = (
        "\n\n".join(render_excerpt(ex) for ex in excerpts)
        if excerpts
        else "_no excerpts available — run with at least one scenario × seed first._"
    )
    return [
        _Section(
            slug="headline",
            title="Headline numbers",
            maps_to="writeup/<sc>_results.txt (opening line)",
            body=headline_table(view),
        ),
        _Section(
            slug="strategy_fingerprint",
            title="Per-tier strategy fingerprint",
            maps_to="writeup/<sc>_design.txt",
            body=strategy_fingerprint_table(view=view, store=store),
        ),
        _Section(
            slug="chain_results",
            title="Chain-level results",
            maps_to="writeup/<sc>_results.txt — Table 1",
            body=chain_results_table(view),
        ),
        _Section(
            slug="per_tier_results",
            title="Per-tier results",
            maps_to="writeup/<sc>_results.txt — Table 2",
            body=per_tier_results_table(view=view, store=store),
        ),
        _Section(
            slug="stability",
            title="Stability check (S1.1 vs S2.3)",
            maps_to="writeup/<sc>_results.txt — Table 3",
            body=stability_table(view),
        ),
        _Section(
            slug="sensitivity",
            title="Sensitivity test",
            maps_to="writeup/<sc>_results.txt — Table 4",
            body=sensitivity_table(view),
        ),
        _Section(
            slug="failure_excerpts",
            title="Failure tick excerpts",
            maps_to="writeup/<sc>_results.txt — reflection paragraph",
            body=excerpts_body,
        ),
        _Section(
            slug="baselines",
            title="Comparison vs baselines",
            maps_to="writeup/<sc>_design.txt — coordination paragraph; "
            "writeup/<sc>_architecture.txt — rejected alternatives",
            body=baselines_comparison_table(
                store=store, batch_id=batch_id, team_id=team_id
            ),
        ),
        _Section(
            slug="reproducibility",
            title="Reproducibility footer",
            maps_to="writeup/<sc>_results.txt (footer)",
            body=reproducibility_table(
                batch_id=batch_id, view=view, git_sha=git_sha
            ),
        ),
    ]


def _render_rundown(
    *, team_id: str, batch_id: str, sections: list[_Section]
) -> str:
    parts: list[str] = [
        f"# {team_id} — batch-run `{batch_id}`",
        "",
        "Cite the tables below directly in your `<sc>_results.txt` and "
        "`<sc>_design.txt`. Each section carries a `<!-- maps to: ... -->` "
        "comment that points to the writeup file it serves.",
        "",
    ]
    for sec in sections:
        anchor = md_html_anchor(maps_to=sec.maps_to)
        parts.append(f"## {sec.title}\n\n{anchor}\n\n{sec.body}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def write_team_rundown(
    *,
    store: RunStore,
    batch_id: str,
    team_id: str,
    out_root: Path,
    git_sha: str | None = None,
) -> Path:
    """Write `rundown.md` + `tables/*.md` for one team. Returns rundown path."""
    sections = _build_sections(
        store=store,
        batch_id=batch_id,
        team_id=team_id,
        git_sha=git_sha,
    )
    team_dir = out_root / batch_id / team_id
    tables_dir = team_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rundown_path = team_dir / "rundown.md"
    rundown_path.write_text(
        _render_rundown(
            team_id=team_id, batch_id=batch_id, sections=sections
        )
    )

    for i, sec in enumerate(sections, start=1):
        snippet = md_section(sec.title, sec.body, level=2)
        (tables_dir / f"{i:02d}__{sec.slug}.md").write_text(snippet)

    return rundown_path


def team_ids_in_batch(
    *, store: RunStore, batch_id: str, exclude: set[str] | None = None
) -> list[str]:
    """List distinct team_ids in a batch-run, with optional exclusions."""
    excl = exclude or set()
    seen: list[str] = []
    for r in store.list_runs(batch_id=batch_id):
        if r.team_id in excl or r.team_id in seen:
            continue
        seen.append(r.team_id)
    return seen


__all__ = [
    "write_team_rundown",
    "team_ids_in_batch",
]
