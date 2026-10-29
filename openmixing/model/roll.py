from __future__ import annotations

from .events import Trigger, new_id

ROLL_CURVES = ("log", "linear")
HIT_BEATS = 0.5


def roll_times(length: float, start_div: float, end_div: float, curve: str) -> list[float]:
    times, t = [], 0.0
    while t < length - 1e-9:
        times.append(t)
        u = t / length
        t += start_div + (end_div - start_div) * u if curve == "linear" else start_div * (end_div / start_div) ** u
    return times


def expand_roll(at: float, length: float, start_div: float, end_div: float, curve: str, gain_db: float, gain_end_db: float,
                kind: str, params: dict) -> list[Trigger]:
    if curve not in ROLL_CURVES:
        raise ValueError(f"curve must be one of {list(ROLL_CURVES)}")
    times = roll_times(length, start_div, end_div, curve)
    return [Trigger(new_id("tg"), kind, at + t, HIT_BEATS, "start", gain_db + (gain_end_db - gain_db) * t / length, dict(params)) for t in times]
