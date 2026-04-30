"""Built-in demand functions.

Each function has signature (period, seed) -> int and is deterministic
given the seed.
"""

from __future__ import annotations

import random


def constant(value: int) -> "DemandFn":
    def fn(period: int, seed: int) -> int:
        return value

    return fn


def step(low: int, high: int, switch_at: int) -> "DemandFn":
    def fn(period: int, seed: int) -> int:
        return low if period < switch_at else high

    return fn


def iid_uniform(low: int, high: int) -> "DemandFn":
    def fn(period: int, seed: int) -> int:
        rng = random.Random(f"iid_uniform|{seed}|{period}")
        return rng.randint(low, high)

    return fn


def seasonal(base: int, amplitude: int, cycle: int) -> "DemandFn":
    import math

    def fn(period: int, seed: int) -> int:
        wave = amplitude * math.sin(2 * math.pi * period / cycle)
        return max(0, int(round(base + wave)))

    return fn


def scripted(values: list[int]) -> "DemandFn":
    def fn(period: int, seed: int) -> int:
        return values[period] if period < len(values) else values[-1]

    return fn


# Type alias for clarity (forward-declared above)
from typing import Callable  # noqa: E402

DemandFn = Callable[[int, int], int]
