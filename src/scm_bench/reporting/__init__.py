"""Markdown report assembly for a batch-run.

Two artifacts per batch-run:
- ``reports/<batch_id>/<team_id>/rundown.md`` — per team, maps 1:1 to
  the write-up template's mandatory tables (`<sc>_results.txt`).
- ``reports/<batch_id>/_aggregate_summary.md`` — one population-level
  Markdown file for operator use.

Tables come from `tables.py`, failure-tick excerpts from `excerpts.py`.
The renderer (`render.py`) is plain f-string composition; no Jinja2.
"""
