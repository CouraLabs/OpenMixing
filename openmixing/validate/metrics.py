from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

from ..core.units import SR
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap, resolve_starts

SILENCE_DB = -60.0
MIN_SILENCE_S = 0.4
LOW_END_DB = -35.0
DRIFT_OK_MS = 25.0


def _beat_times(c, tm: TempoMap, start: float) -> np.ndarray:
    return start + np.asarray(tm.beat_to_sec(np.arange(0, np.floor(c.length_beats) + 1, 1.0)))


def beat_drift(model: SetModel, info) -> list[dict]:
    starts = resolve_starts(model, lambda t: float(info(t)["bpm"]))
    maps = {c.id: TempoMap(c, float(info(c.track)["bpm"])) for c in model.clips}
    beats = {c.id: _beat_times(c, maps[c.id], starts[c.id]) for c in model.clips}
    out = []
    for i, a in enumerate(model.clips):
        for b in model.clips[i + 1:]:
            lo, hi = max(beats[a.id][0], beats[b.id][0]), min(beats[a.id][-1], beats[b.id][-1])
            if hi - lo < 1.0:
                continue
            ta = beats[a.id][(beats[a.id] >= lo) & (beats[a.id] <= hi)]
            tb = beats[b.id]
            d = np.array([np.min(np.abs(tb - t)) for t in ta]) * 1000.0
            out.append({"clips": [a.id, b.id], "overlap_sec": [round(float(lo), 2), round(float(hi), 2)],
                        "max_drift_ms": round(float(d.max()), 1), "mean_drift_ms": round(float(d.mean()), 1),
                        "ok": bool(d.max() < DRIFT_OK_MS)})
    return out


def _lufs(x: np.ndarray) -> float | None:
    if x.shape[1] < int(0.5 * SR):
        return None
    v = pyln.Meter(SR).integrated_loudness(x.T)
    return None if not np.isfinite(v) else round(float(v), 2)


def loudness_continuity(x: np.ndarray, t0: float, rows: list[dict], window: float = 8.0) -> list[dict]:
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            lo, hi = max(a["start_sec"], b["start_sec"]), min(a["end_sec"], b["end_sec"])
            if hi - lo < 1.0:
                continue
            seg = lambda s, e: x[:, max(int((s - t0) * SR), 0): max(int((e - t0) * SR), 0)]
            before, after = _lufs(seg(lo - window, lo)), _lufs(seg(hi, hi + window))
            out.append({"clips": [a["clip"], b["clip"]], "before_lufs": before, "after_lufs": after,
                        "jump_lu": None if before is None or after is None else round(after - before, 2)})
    return out


def silences(x: np.ndarray, t0: float, rows: list[dict]) -> list[dict]:
    blk = SR // 20
    n = x.shape[1] // blk
    if n == 0:
        return []
    rms = np.sqrt((x[:, : n * blk].reshape(2, n, blk) ** 2).mean(axis=(0, 2)) + 1e-12)
    quiet = 20 * np.log10(rms) < SILENCE_DB
    out, i = [], 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        a, b = t0 + i * blk / SR, t0 + j * blk / SR
        if b - a >= MIN_SILENCE_S:
            covered = any(r["start_sec"] < b and r["end_sec"] > a for r in rows)
            out.append({"start_sec": round(a, 2), "end_sec": round(b, 2), "planned": not covered})
        i = j
    return out


def low_end_overlap(low_db: dict, block_sec: float, t0: float) -> dict:
    if len(low_db) < 2:
        return {"overlap_sec": 0.0, "regions": [], "threshold_db": LOW_END_DB}
    ids = list(low_db)
    n = min(len(v) for v in low_db.values())
    on = np.array([np.asarray(low_db[i])[:n] > LOW_END_DB for i in ids])
    both = on.sum(axis=0) >= 2
    regions, i = [], 0
    while i < n:
        if not both[i]:
            i += 1
            continue
        j = i
        while j < n and both[j]:
            j += 1
        regions.append({"start_sec": round(t0 + i * block_sec, 2), "end_sec": round(t0 + j * block_sec, 2),
                        "clips": [c for k, c in enumerate(ids) if on[k, i:j].any()]})
        i = j
    return {"overlap_sec": round(float(both.sum() * block_sec), 2), "regions": regions[:8], "threshold_db": LOW_END_DB}
