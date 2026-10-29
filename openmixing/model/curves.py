from __future__ import annotations

import math

from .lane import Lane

CURVES = ("linear", "equal_power", "log", "s_curve", "power", "sharp_cut")


def shape(curve: str, k: float, rising: bool):
    def base(p: float) -> float:
        if curve == "equal_power":
            return math.sin(p * math.pi / 2)
        if curve == "log":
            return math.log10(1 + 9 * p)
        if curve == "s_curve":
            return 0.5 - 0.5 * math.cos(math.pi * p)
        if curve == "power":
            return p ** k
        return p
    if curve in ("equal_power", "log") and not rising:
        return lambda p: 1 - base(1 - p)
    return base


def write(lane: Lane, at: float, value: float, ramp: float, curve: str = "linear", k: float = 2.0) -> None:
    if curve == "linear" or ramp <= 0:
        lane.set(at, value, ramp)
    elif curve == "sharp_cut":
        lane.set(at + ramp, value, 0.0)
    else:
        v0 = lane.value_at(at)
        sh = shape(curve, k, value >= v0)
        lane.curve(at, ramp, lambda p: v0 + (value - v0) * sh(p), steps_per_beat=4)
