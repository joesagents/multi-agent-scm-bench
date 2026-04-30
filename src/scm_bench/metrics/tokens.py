"""Aggregate AgentDecision.tokens_used into per-tier and chain totals.

Reference starters always emit 0; the LLM baseline emits real counts.
The aggregator sums per role and totals the chain so the composite
score and the per-team tables have one canonical number.
"""

from __future__ import annotations

from scm_bench.engine.state import TIER_NAMES
from scm_bench.runner.harness import RunRecord


def aggregate_tokens(record: RunRecord) -> dict[str, int]:
    """Per-role and chain-level token totals.

    Returns a dict shaped like::

        {"retailer": 0, "wholesaler": 0, "distributor": 0, "factory": 0,
         "chain": 0}
    """
    totals: dict[str, int] = {role: 0 for role in TIER_NAMES}
    for tick in record.ticks:
        for role, decision in tick.decisions.items():
            totals[role] += int(decision.tokens_used)
    totals["chain"] = sum(totals[role] for role in TIER_NAMES)
    return totals
