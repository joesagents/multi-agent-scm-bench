"""Core supply chain bench metrics.

CV and bullwhip are ported verbatim from the historical baseline
implementation. Reproducibility of previously published baseline numbers
depends on these formulas matching exactly.
"""

from collections.abc import Sequence

import numpy as np


def coefficient_of_variation(data: Sequence[int | float]) -> float:
    """CV = std/mean (sample SD, ddof=1).

    Returns 0.0 if data has fewer than 2 points or mean is 0.
    """
    if len(data) < 2:
        return 0.0
    mean = float(np.mean(data))
    if mean == 0:
        return 0.0
    return float(np.std(data, ddof=1) / mean)


def calculate_bullwhip(orders: Sequence[int], demand: Sequence[int]) -> float:
    """Bullwhip as CV difference per Jin, DeHoratius & Schmidt (2017).

    bullwhip = CV(orders placed) - CV(orders received)

    Positive  → amplification.
    Zero      → pass-through.
    Negative  → smoothing.
    """
    return coefficient_of_variation(orders) - coefficient_of_variation(demand)


def fill_rate(shipped: Sequence[int], orders_received: Sequence[int]) -> float:
    """Fraction of received orders that were shipped (1.0 if no orders received)."""
    total_shipped = sum(shipped)
    total_received = sum(orders_received)
    if total_received <= 0:
        return 1.0
    return total_shipped / total_received


def average(values: Sequence[int | float]) -> float:
    """Arithmetic mean; 0.0 on empty input."""
    if not values:
        return 0.0
    return float(np.mean(values))
