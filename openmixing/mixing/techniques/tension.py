from __future__ import annotations

import numpy as np

TENSION_MAP = {"filter": (0.0, 0.85), "eq_low": (0.0, -60.0), "volume": (0.85, 1.0), "riser_db": (-18.0, -3.0), "roll_db": (-24.0, -6.0)}
LOOP_ROLL_MIN_LEVEL = 0.5


def mapped(name: str, level: float) -> float:
    lo, hi = TENSION_MAP[name]
    return lo + (hi - lo) * float(level)


def level_fn(points: list[list[float]] | None, start: float, length: float, peak: float):
    if not points:
        return lambda p: peak * (1 - np.cos(np.pi * p)) / 2
    xs, ys = zip(*sorted(points))
    return lambda p: float(np.interp(start + p * length, xs, ys))
