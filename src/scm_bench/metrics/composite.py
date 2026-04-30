"""Composite benchmark score — agent-facing definition.

Three components, all lower-is-better, normalized against the echo
baseline (Mirror agent), then combined with a geometric mean.

    bullwhip_ratio = variance(factory orders) / variance(consumer demand)
    total_cost     = chain-level cumulative holding + backlog cost
    tokens_used    = sum of AgentDecision.tokens_used across all ticks

    normalized_X    = (your_X + epsilon) / (baseline_X + epsilon)
    composite_score = (n_bullwhip * n_cost * n_tokens) ** (1/3)

Epsilon=1.0 lets the Mirror baseline (zero tokens, possibly zero
bullwhip on perfectly stable demand) participate without divide-by-zero.

Note: the engine's `bullwhip` field is a CV difference (Jin/DeHoratius/
Schmidt 2017), kept for v1 parity. The user-facing `bullwhip_ratio`
here is the variance ratio described in the metrics primer; both
coexist intentionally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPSILON: float = 1.0


def bullwhip_ratio_variance(
    orders: Sequence[int | float],
    demand: Sequence[int | float],
) -> float:
    """Variance ratio of upstream orders to consumer demand.

    Returns 1.0 when demand variance is zero (perfectly flat demand —
    no signal to amplify).
    """
    if len(orders) < 2 or len(demand) < 2:
        return 1.0
    var_orders = float(np.var(orders, ddof=1))
    var_demand = float(np.var(demand, ddof=1))
    if var_demand <= 0:
        return 1.0
    return var_orders / var_demand


@dataclass(frozen=True)
class ScoreComponents:
    """The three scalar inputs to the composite score, plus the score."""

    bullwhip_ratio: float
    total_cost: float
    tokens_used: float

    def as_dict(self) -> dict[str, float]:
        return {
            "bullwhip_ratio": self.bullwhip_ratio,
            "total_cost": self.total_cost,
            "tokens_used": self.tokens_used,
        }


@dataclass(frozen=True)
class CompositeBreakdown:
    """Per-component normalised values + the geometric-mean composite."""

    normalized_bullwhip: float
    normalized_cost: float
    normalized_tokens: float
    composite_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "normalized_bullwhip": self.normalized_bullwhip,
            "normalized_cost": self.normalized_cost,
            "normalized_tokens": self.normalized_tokens,
            "composite_score": self.composite_score,
        }


def normalize_components(
    your: ScoreComponents,
    baseline: ScoreComponents,
    *,
    epsilon: float = EPSILON,
) -> CompositeBreakdown:
    """Per-component (your+ε)/(baseline+ε), then geometric mean."""
    n_bullwhip = (your.bullwhip_ratio + epsilon) / (baseline.bullwhip_ratio + epsilon)
    n_cost = (your.total_cost + epsilon) / (baseline.total_cost + epsilon)
    n_tokens = (your.tokens_used + epsilon) / (baseline.tokens_used + epsilon)
    composite = float((n_bullwhip * n_cost * n_tokens) ** (1.0 / 3.0))
    return CompositeBreakdown(
        normalized_bullwhip=n_bullwhip,
        normalized_cost=n_cost,
        normalized_tokens=n_tokens,
        composite_score=composite,
    )


def components_from_record(
    *,
    factory_orders: Sequence[int],
    consumer_demand: Sequence[int],
    total_cost: float,
    tokens_used: float,
) -> ScoreComponents:
    """Bundle the three composite inputs for one (team, scenario, seed) cell."""
    return ScoreComponents(
        bullwhip_ratio=bullwhip_ratio_variance(factory_orders, consumer_demand),
        total_cost=total_cost,
        tokens_used=tokens_used,
    )
