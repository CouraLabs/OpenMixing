from __future__ import annotations

import numpy as np

from .. import candidates


def window_span(f: dict) -> tuple[int, int]:
    n, ph = f["n_bars"], f["phrase"]
    ent, ext = candidates.find_windows(f, "entry")[0], candidates.find_windows(f, "exit")[0]
    a = min(int(ent["start_bar"]) // ph * ph, max(n - 8, 0))
    b = min(int(ext["end_bar"]), n)
    return (a, b) if b - a >= 8 else (0, n)


def slice_energy(f: dict, a: float, b: float) -> float:
    curve = np.asarray(f["energy"], float)
    whole = float(curve.mean()) if len(curve) else 0.0
    part = curve[int(a):max(int(b), int(a) + 1)]
    if not whole or not len(part):
        return float(f["e"]["energy"])
    return float(min(max(f["e"]["energy"] * part.mean() / whole, 1.0), 10.0))


def choices(f: dict, start: int, end: int, min_play: int) -> list[int]:
    ph, avail = f["phrase"], end - start
    lo = min(min_play, avail)
    return sorted({avail, *range(-(-lo // ph) * ph, avail + 1, ph)})


def fit(opts: list[list[int]], bpms: list[float], target_sec: float) -> list[int]:
    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for o, bpm in zip(opts, bpms):
        nxt: dict[int, tuple[float, list[int]]] = {}
        top = max(o)
        for s, (pen, path) in states.items():
            for p in o:
                k = s + int(round(p * 240.0 / bpm))
                v = (pen + (top - p) * 1e-3, path + [p])
                if k not in nxt or v[0] < nxt[k][0]:
                    nxt[k] = v
        states = nxt
    best = min(states, key=lambda k: (abs(k - target_sec), states[k][0]))
    return states[best][1]


def trim(facts: list[dict], bpms: list[float], target_sec: float, min_play: int) -> list[dict]:
    spans = [window_span(f) for f in facts]
    opts = [choices(f, a, b, min_play) for f, (a, b) in zip(facts, spans)]
    plays = fit(opts, bpms, target_sec)
    return [{"start_bar": a, "bars": p, "avail": b - a, "energy": slice_energy(f, a, a + p)} for f, (a, b), p in zip(facts, spans, plays)]
