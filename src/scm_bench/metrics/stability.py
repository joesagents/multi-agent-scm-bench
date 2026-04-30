"""Stability check — does a team's stable-vs-disrupted ranking direction hold?

The write-up template requires a stability table comparing each team's
chain-level metrics on the stable (s1.1) vs disrupted (s2.3) scenarios,
with an auto-computed "direction held?" boolean: a team ranks in the
same population quartile on both scenarios.

This module exposes the small helpers the report renderer needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def quartile(value: float, distribution: Sequence[float]) -> int:
    """Return the 1-indexed quartile of ``value`` within ``distribution``.

    Lower values land in lower quartiles. Used for composite scores
    where lower is better, so quartile 1 = strongest performers.
    """
    if not distribution:
        return 0
    if len(distribution) == 1:
        return 1
    edges = np.quantile(np.asarray(distribution, dtype=float), [0.25, 0.5, 0.75])
    if value <= edges[0]:
        return 1
    if value <= edges[1]:
        return 2
    if value <= edges[2]:
        return 3
    return 4


def direction_held(
    *,
    team_score_stable: float,
    team_score_disrupted: float,
    population_scores_stable: Sequence[float],
    population_scores_disrupted: Sequence[float],
) -> bool:
    """True iff the team is in the same quartile on both scenarios.

    "Direction-of-improvement" check: a team whose
    relative performance survives a regime change shows robust strategy
    rather than overfitting to one demand pattern.
    """
    q_stable = quartile(team_score_stable, population_scores_stable)
    q_disrupted = quartile(team_score_disrupted, population_scores_disrupted)
    return q_stable != 0 and q_stable == q_disrupted
