from __future__ import annotations

import numpy as np

from ..core.errors import OpenMixingError

PRESETS = {
    "warm_up_peak_cooldown": [[0, 0.2], [0.15, 0.35], [0.5, 1.0], [0.75, 0.9], [1, 0.25]],
    "build": [[0, 0.1], [1, 1.0]],
    "plateau": [[0, 0.8], [1, 0.8]],
    "wave": [[0, 0.3], [0.25, 0.9], [0.5, 0.3], [0.75, 0.9], [1, 0.3]],
}


def points(arc, lo: float, hi: float, field: str) -> list[list[float]]:
    if isinstance(arc, str):
        if arc not in PRESETS:
            raise OpenMixingError("invalid_value", f"unknown arc preset {arc!r}", field, f"use one of {list(PRESETS)} or [[t_frac, value], ...]")
        return [[t, lo + v * (hi - lo)] for t, v in PRESETS[arc]]
    pts = sorted([float(t), float(v)] for t, v in arc)
    if len(pts) < 2 or any(not 0 <= t <= 1 for t, _ in pts):
        raise OpenMixingError("invalid_value", "an arc needs at least 2 points with t_frac in 0..1", field, "pass [[0, v0], [1, v1]]")
    return pts


def at(pts: list[list[float]], frac: float) -> float:
    return float(np.interp(min(max(frac, 0.0), 1.0), [p[0] for p in pts], [p[1] for p in pts]))


def norm(pts: list[list[float]], frac: float) -> float:
    vs = [p[1] for p in pts]
    lo, hi = min(vs), max(vs)
    return (at(pts, frac) - lo) / (hi - lo) if hi > lo else 0.5


def make(items: list[dict], energy_arc, tempo_arc, key_arc, tempo_follows: bool, key_follows: bool, base_key: int) -> dict:
    es, bs = [i["energy"] for i in items], [i["bpm"] for i in items]
    e = points(energy_arc, min(es), max(es), "energy_arc") if energy_arc is not None else None
    if tempo_arc is None and tempo_follows and e:
        tempo_arc = [[t, (v - min(es)) / max(max(es) - min(es), 1e-9)] for t, v in e]
        tempo_arc = [[t, min(bs) + v * (max(bs) - min(bs))] for t, v in tempo_arc]
    t = points(tempo_arc, min(bs), max(bs), "tempo_arc") if tempo_arc is not None else None
    if key_arc is None and key_follows and e:
        key_arc = [[tt, base_key + 3 * norm(e, tt)] for tt, _ in e]
    k = points(key_arc, 1, 12, "key_arc") if key_arc is not None else None
    return {"energy": e, "tempo": t, "key": k}
