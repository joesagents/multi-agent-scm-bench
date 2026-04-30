"""Tiny Markdown helpers — table rows, formatting.

Kept dependency-free so the report renderer can run on any Python the
v2 SDK supports.
"""

from __future__ import annotations

from collections.abc import Sequence


def fmt_float(x: float | None, *, places: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{places}f}"


def fmt_int(x: int | None) -> str:
    if x is None:
        return "—"
    return f"{x:d}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a Markdown table; empty rows render as a single em-dash row."""
    if not rows:
        rows = [["—"] * len(headers)]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(c) for c in r) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def md_section(title: str, body: str, *, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body}\n"


def md_html_anchor(*, maps_to: str) -> str:
    """An HTML comment that ties this section to a writeup file."""
    return f"<!-- maps to: {maps_to} -->"
