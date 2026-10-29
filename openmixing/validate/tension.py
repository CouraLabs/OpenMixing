from __future__ import annotations

import librosa
import numpy as np

from ..analysis.confidence import combine, unit
from ..analysis.tension import TENSION_WEIGHTS, scaled

SHAPES = ("rising", "falling", "build_release", "plateau")
MOVING = 1e-3


def _chunk(y: np.ndarray, sr: int, win: float) -> list[list[float]]:
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    per = max(int(round(win * sr / hop)), 1)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop)
    cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S[freqs > 2500]))[0]
    tot = np.maximum(S.sum(0), 1e-12)
    bass = S[freqs < 150].sum(0) / tot
    out = []
    for a in range(0, S.shape[1] - per + 1, per):
        sl = slice(a, a + per)
        out.append([onset[sl].mean(), cen[sl].mean(), np.log10(tot[sl].mean() + 1e-12), 1 - bass[sl].mean(), flat[sl].mean() * onset[sl].mean()])
    return out


def rendered_tension(y: np.ndarray, sr: int, win: float = 1.0, chunk_windows: int = 60) -> np.ndarray:
    step = int(round(win * sr)) * chunk_windows
    rows = [r for a in range(0, len(y), step) for r in _chunk(y[a : a + step], sr, win)]
    if len(rows) < 2:
        return np.zeros(len(rows))
    f = np.array(rows).T
    energy = scaled(f[2], 1.0)
    rise = np.clip(energy - energy[np.maximum(np.arange(len(energy)) - 4, 0)], 0, 1)
    comps = {"density": scaled(f[0], 1.0), "brightness": scaled(f[1], 300.0), "rise": rise, "bass_absence": scaled(f[3], 0.05),
             "noise": scaled(f[4], 0.05)}
    return sum(TENSION_WEIGHTS[k] * v for k, v in comps.items())


def tension_target(name: str, n: int) -> np.ndarray:
    u = np.linspace(0, 1, n)
    return {"rising": u, "falling": 1 - u, "plateau": np.full(n, 0.5),
            "build_release": np.where(u < 0.8, u / 0.8, 0.1)}[name]


def points_target(points: list, times: np.ndarray) -> np.ndarray:
    p = sorted(points)
    return np.interp(times, [a for a, _ in p], [b for _, b in p])


def tension_match(curve: np.ndarray, t0: float, win: float, spec: dict) -> dict:
    times = t0 + win * np.arange(len(curve))
    a, b = spec.get("start_sec", times[0] if len(times) else 0.0), spec.get("end_sec", times[-1] + win if len(times) else 0.0)
    m = (times >= a) & (times < b)
    got = curve[m]
    base = {"window_sec": [round(float(a), 2), round(float(b), 2)], "target": spec.get("shape") or "points", "windows": int(m.sum())}
    if len(got) < 3:
        return {"correlation": None, "agreement": None, "max_error": None, "confidence": 0.0, **base}
    tgt = tension_target(spec["shape"], len(got)) if "shape" in spec else points_target(spec["points"], times[m])
    corr = 0.0 if np.ptp(got) < 1e-9 or np.ptp(tgt) < 1e-9 else float(np.corrcoef(got, tgt)[0, 1])
    dg, dt = np.diff(got), np.diff(tgt)
    moving = np.abs(dt) > MOVING
    agree = float((np.sign(dg[moving]) == np.sign(dt[moving])).mean()) if moving.any() else 1.0
    conf, comps = combine({"contrast": unit(float(np.percentile(got, 90) - np.percentile(got, 10)), 0, 0.4), "windows": unit(len(got), 8, 32)})
    return {"correlation": round(corr, 3), "agreement": round(agree, 3), "max_error": round(float(np.abs(got - tgt).max()), 3),
            "confidence": conf, "confidence_components": comps, **base}
