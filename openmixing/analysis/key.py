from __future__ import annotations

import librosa
import numpy as np

from .analyze import ASR
from .confidence import attach, unit

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
PROFILES = {
    "krumhansl": ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "temperley": ([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                  [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
}


def camelot(pc: int, minor: bool) -> str:
    return f"{CAMELOT_MAJOR[(pc + 3) % 12] if minor else CAMELOT_MAJOR[pc]}{'A' if minor else 'B'}"


def key_name(pc: int, minor: bool) -> str:
    return f"{NOTES[pc]} {'minor' if minor else 'major'}"


def rank_keys(chroma_mean: np.ndarray) -> list[dict]:
    res = []
    for pc in range(12):
        for minor in (False, True):
            r = np.mean([np.corrcoef(np.roll(p[int(minor)], pc), chroma_mean)[0, 1] for p in PROFILES.values()])
            res.append({"key": key_name(pc, minor), "camelot": camelot(pc, minor), "score": round(float(r), 4),
                        "_pc": pc, "_minor": minor})
    return sorted(res, key=lambda d: -d["score"])


def key_analysis(y: np.ndarray, dur: float) -> dict:
    yh = librosa.effects.harmonic(y, margin=3)
    tuning = float(librosa.estimate_tuning(y=yh, sr=ASR))
    chroma = librosa.feature.chroma_cqt(y=yh, sr=ASR, tuning=tuning, hop_length=2048)
    ranked = rank_keys(chroma.mean(1))
    best, second = ranked[0], ranked[1]
    margin = best["score"] - second["score"]
    out = {
        "key": best["key"], "camelot": best["camelot"], "margin": round(margin, 4),
        "correlation": best["score"], "tuning_cents": round(tuning * 100, 1),
        "candidates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in ranked[:4]],
        "chroma_mean": [round(float(v), 4) for v in chroma.mean(1) / max(chroma.mean(1).max(), 1e-9)],
    }
    n = chroma.shape[1]
    seg = max(n // 6, 1)
    track = []
    for i in range(0, n - seg + 1, seg):
        r = rank_keys(chroma[:, i : i + seg].mean(1))[0]
        track.append({"start_sec": round(i * 2048 / ASR, 1), "key": r["key"], "camelot": r["camelot"],
                      "score": r["score"]})
    out["key_over_time"] = track
    out["key_stable"] = len({t["camelot"] for t in track}) <= 2
    agree = float(np.mean([t["camelot"] == best["camelot"] for t in track])) if track else 0.0
    return attach(out, {"margin": unit(margin, 0, 0.125), "correlation": unit(best["score"], 0.4, 0.9), "agreement_over_time": agree})


def harmonic_neighbours(cam: str) -> dict:
    n, m = int(cam[:-1]), cam[-1]
    wrap = lambda k: (k - 1) % 12 + 1
    other = "A" if m == "B" else "B"
    return {"perfect": [cam], "smooth": [f"{wrap(n + 1)}{m}", f"{wrap(n - 1)}{m}"], "relative": [f"{n}{other}"],
            "energy_boost": [f"{wrap(n + 2)}{m}", f"{wrap(n + 7)}{m}"]}


def camelot_relation(a: str, b: str) -> tuple[int, str]:
    na, ma, nb, mb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    if a == b:
        return 0, "same key"
    if na == nb:
        return 1, "relative major/minor"
    step = min((na - nb) % 12, (nb - na) % 12)
    if ma == mb:
        return step, {1: "adjacent on the wheel (smooth)", 2: "two steps (energy shift)",
                      7: "perfect-fifth-ish energy boost"}.get(step, f"{step} steps apart")
    return step + 1, f"{step} steps and mode change"
