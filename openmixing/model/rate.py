from __future__ import annotations

import numpy as np

from .clip import Clip
from .events import RateEvent

KINDS = ("spinback", "reverse", "brake", "custom")
SPINBACK_POWER = 2.5
STEPS = 16


def preset(kind: str) -> list[list[float]]:
    if kind == "spinback":
        return [[i / STEPS, 1.0 - (i / STEPS) ** SPINBACK_POWER] for i in range(STEPS + 1)]
    if kind == "reverse":
        return [[0.0, -1.0], [1.0, -1.0]]
    if kind == "brake":
        return [[0.0, 1.0], [1.0, 0.0]]
    raise ValueError(f"kind {kind!r} has no preset curve: give a curve of [u, speed] points")


def check_curve(pts: list[list[float]], lo: float, hi: float) -> None:
    if len(pts) < 2 or any(len(p) != 2 for p in pts):
        raise ValueError("a rate curve needs at least two [u, speed] points")
    us = [p[0] for p in pts]
    if us[0] != 0 or us[-1] != 1 or any(b <= a for a, b in zip(us, us[1:])):
        raise ValueError("rate curve u values must increase strictly from 0 to 1")
    if any(not lo <= p[1] <= hi for p in pts):
        raise ValueError(f"rate curve speeds must be within [{lo:g}, {hi:g}]")


def speed_at(ev: RateEvent, u: np.ndarray) -> np.ndarray:
    xs, ys = zip(*ev.curve)
    return np.interp(u, xs, ys)


def advance(ev: RateEvent, run: float | None = None) -> float:
    run = ev.duration_beats if run is None else min(run, ev.duration_beats)
    u_end = run / ev.duration_beats
    us = np.array(sorted({0.0, u_end, *[p[0] for p in ev.curve if 0 < p[0] < u_end]}))
    return float(np.trapezoid(speed_at(ev, us), us)) * ev.duration_beats


def net_shift(clip: Clip) -> float:
    return sum(advance(e, clip.length_beats - e.beat) - min(e.duration_beats, clip.length_beats - e.beat)
               for e in clip.rate if not e.slip and e.beat < clip.length_beats)
