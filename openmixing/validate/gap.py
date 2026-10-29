from __future__ import annotations

import numpy as np

from ..model.tempo import TempoMap

BLOCK_SEC = 0.025
REF_SEC = 2.0
GAP_DB = 12.0


def silent_gaps(y: np.ndarray, sr: int, t0: float) -> list[dict]:
    blk = int(BLOCK_SEC * sr)
    n = len(y) // blk
    p = (y[: n * blk].reshape(n, blk) ** 2).mean(1)
    k, thr = int(REF_SEC / BLOCK_SEC), 10 ** (-GAP_DB / 10)
    out, i = [], k
    while i < n:
        ref = p[i - k : i].mean()
        if ref <= 1e-10 or p[i] >= ref * thr:
            i += 1
            continue
        j = i
        while j < n and p[j] < ref * thr:
            j += 1
        out.append({"start_sec": round(t0 + i * BLOCK_SEC, 3), "end_sec": round(t0 + j * BLOCK_SEC, 3)})
        i = j
    return out


def gap_beats(gaps: list[dict], model, info, rows: list[dict]) -> list[dict]:
    clips = {c.id: c for c in model.clips}
    out = []
    for g in gaps:
        r = next((r for r in rows if r["start_sec"] <= g["start_sec"] < r["end_sec"]), None)
        if r is None:
            continue
        tm = TempoMap(clips[r["clip"]], float(info(r["track"])["bpm"]))
        beats = float(tm.sec_to_beat(g["end_sec"] - r["start_sec"]) - tm.sec_to_beat(g["start_sec"] - r["start_sec"]))
        out.append({**g, "beats": round(beats, 2)})
    return out
