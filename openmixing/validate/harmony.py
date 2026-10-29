from __future__ import annotations

import numpy as np

from ..analysis.confidence import combine, unit
from ..analysis.key import camelot_relation, rank_keys

ACTIVE_DB = -50.0
DISSONANT = 0.3
TOP_PC = 4
PEAK_FRAC = 0.5
ROUGH = (0.0, 1.0, 0.4, 0.2, 0.15, 0.1, 0.6)


def _top(c: np.ndarray) -> dict[int, float]:
    return {int(p): float(c[p]) for p in np.argsort(c)[-TOP_PC:] if c[p] >= PEAK_FRAC * c.max()}


def dissonance(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = _top(a), _top(b)
    mass = rough = 0.0
    for i, wi in ta.items():
        for j, wj in tb.items():
            w = min(wi, wj)
            d = (i - j) % 12
            mass, rough = mass + w, rough + w * ROUGH[min(d, 12 - d)]
    return rough / mass if mass > 0 else 0.0


def harmonic_clash(chroma: dict, chroma_db: dict, block_sec: float, t0: float) -> dict:
    base = {"method": "chroma_roughness"}
    ids = list(chroma)
    if len(ids) < 2:
        return {"overlap_sec": 0.0, "mean_dissonance": None, "worst_segments": [], "key_distance_max": None, "status": "not_applicable",
                "confidence": 0.0, **base}
    n = min(c.shape[1] for c in chroma.values())
    act = np.array([np.asarray(chroma_db[i])[:n] > ACTIVE_DB for i in ids])
    pairs = [(a, b) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    rows, keys = [], []
    for a, b in pairs:
        blocks = np.flatnonzero(act[a] & act[b])
        if not len(blocks):
            continue
        ca, cb = (chroma[ids[k]][:, blocks].astype(np.float32) for k in (a, b))
        d = [dissonance(ca[:, k], cb[:, k]) for k in range(len(blocks))]
        rows += [(int(j), ids[a], ids[b], v) for j, v in zip(blocks, d)]
        ra, rb = rank_keys(ca.mean(1)), rank_keys(cb.mean(1))
        keys.append((camelot_relation(ra[0]["camelot"], rb[0]["camelot"])[0], min(ra[0]["score"], rb[0]["score"]),
                     float(min(np.mean(np.asarray(chroma_db[ids[a]])[blocks]), np.mean(np.asarray(chroma_db[ids[b]])[blocks])))))
    if not rows:
        return {"overlap_sec": 0.0, "mean_dissonance": None, "worst_segments": [], "key_distance_max": None, "status": "not_applicable",
                "confidence": 0.0, **base}
    vals = np.array([r[3] for r in rows])
    worst = sorted(rows, key=lambda r: -r[3])[:3]
    conf, comps = combine({"tonal_clarity": unit(min(k[1] for k in keys), 0.4, 0.9), "level": unit(min(k[2] for k in keys), -60, -30)})
    return {"overlap_sec": round(len(set(r[0] for r in rows)) * block_sec, 2), "mean_dissonance": round(float(vals.mean()), 3),
            "worst_segments": [{"start_sec": round(t0 + j * block_sec, 2), "end_sec": round(t0 + (j + 1) * block_sec, 2), "clips": [a, b],
                                "dissonance": round(float(v), 3)} for j, a, b, v in worst],
            "key_distance_max": max(k[0] for k in keys), "status": "measured", "confidence": conf, "confidence_components": comps, **base}
