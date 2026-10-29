from __future__ import annotations

import numpy as np

from .analyze import ASR
from .confidence import attach, unit
from .spectral import _db

FPS = ASR / 512
GAP_DB = 12.0
TENSION_WEIGHTS = {"density": 0.25, "brightness": 0.2, "rise": 0.25, "bass_absence": 0.15, "noise": 0.15}


def scaled(v: np.ndarray, floor: float) -> np.ndarray:
    lo = np.percentile(v, 5)
    return np.clip((v - lo) / max(np.percentile(v, 95) - lo, floor), 0, 1)


def tension_block(bf: dict, st: dict) -> dict:
    n = bf["n_bars"]
    if not st["sections"]:
        return {"curve": [], "components": {}, "weights": TENSION_WEIGHTS, "peaks": [], "confidence": 0.0,
                "confidence_components": {"bars_available": round(n / 8, 3)}}
    X = bf["X"]
    energy, bass = np.array(st["energy_curve"]), np.array(st["bass_curve"])
    comps = {"density": scaled(np.array(st["onset_curve"]), 2.0), "brightness": scaled(X[:, 80], 300.0),
             "rise": np.clip(energy - energy[np.maximum(np.arange(n) - 4, 0)], 0, 1), "bass_absence": 1 - bass,
             "noise": scaled(X[:, 81] * X[:, 82], 0.02)}
    curve = sum(TENSION_WEIGHTS[k] * v for k, v in comps.items())
    peaks = []
    for s in st["sections"]:
        a = s["start_bar"]
        if s["label"] != "drop" or a < 1:
            continue
        lo = max(a - 8, 0)
        peaks.append({"bar": lo + int(np.argmax(curve[lo:a])), "kind": "buildup_peak"})
        if curve[a - 1] - curve[a] > 0.4:
            peaks.append({"bar": a, "kind": "release"})
    block = {"curve": [round(float(v), 3) for v in curve], "components": {k: [round(float(x), 3) for x in v] for k, v in comps.items()},
             "weights": TENSION_WEIGHTS, "peaks": sorted(peaks, key=lambda p: p["bar"])}
    return attach(block, {"bars_available": unit(n, 8, 32), "contrast": unit(float(np.percentile(curve, 90) - np.percentile(curve, 10)), 0, 0.5)})


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def lead_ins(S: np.ndarray, secs: list[dict], fd: float, bar_sec: float, beat_sec: float) -> None:
    db = _db((S ** 2).sum(0))
    for i, s in enumerate(secs):
        a = s["start_bar"]
        if a < 2 or not (s["label"] == "drop" or (s["label"] == "groove" and secs[i - 1]["label"] == "intro")):
            continue
        w = db[int((fd + (a - 2) * bar_sec) * FPS):int((fd + a * bar_sec) * FPS)]
        if w.size < 4:
            continue
        ref = _db(np.mean(10 ** (w[:w.size // 2] / 10)))
        depth = float(ref - w.min())
        s["lead_in"] = attach({"gap_beats": round(float(_longest_run(w < ref - GAP_DB) / FPS / beat_sec), 3), "gap_depth_db": round(depth, 1)},
                              {"depth": unit(depth, GAP_DB, 40)})
