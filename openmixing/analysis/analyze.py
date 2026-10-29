from __future__ import annotations

import librosa
import numpy as np

AUDIO_EXT = {".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".aac", ".ogg", ".opus", ".alac"}
ASR, HOP = 22050, 256
TARGET_LUFS = -14.0
ONSET_LATENCY = 0.013


def _grid_score(env: np.ndarray, t0: float, period: float, n: int) -> float:
    t = (t0 + np.arange(n) * period) * ASR / HOP
    ok = (t >= 0) & (t < len(env) - 1)
    return float(np.interp(t[ok], np.arange(len(env)), env).sum()) if ok.any() else -1.0


def _fold(bpm: float) -> float:
    while bpm < 85:
        bpm *= 2
    while bpm >= 175:
        bpm /= 2
    return bpm


def _score(env: np.ndarray, t0: float, period: float, lo: float = 0.0, hi: float | None = None) -> float:
    hi = len(env) * HOP / ASR if hi is None else hi
    k = np.arange(int(np.ceil((lo - t0) / period)), int(np.ceil((hi - t0) / period)))
    t = (t0 + k * period) * ASR / HOP
    t = t[(t >= 0) & (t < len(env) - 1)]
    return float(np.interp(t, np.arange(len(env)), env).sum()) if len(t) else -1e9


def beat_grid(y: np.ndarray) -> dict:
    env = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP)
    env = np.maximum(env - np.convolve(env, np.ones(43) / 43, mode="same"), 0)
    dur = len(y) / ASR
    seed = float(np.atleast_1d(librosa.beat.beat_track(onset_envelope=env, sr=ASR, hop_length=HOP, start_bpm=124)[0])[0])
    p0 = 60.0 / _fold(seed)

    best = (-1e9, 0.0, p0)
    for pf in np.linspace(0.985, 1.015, 61):
        p = p0 * pf
        for t0 in np.arange(0, p, 0.006):
            sc = _score(env, t0, p)
            if sc > best[0]:
                best = (sc, t0, p)
    _, t0, p = best
    for pf in np.linspace(0.9985, 1.0015, 31):
        for d in np.arange(-0.008, 0.0081, 0.001):
            sc = _score(env, t0 + d, p * pf)
            if sc > best[0]:
                best = (sc, t0 + d, p * pf)
    _, t0, p = best

    snap = 60.0 / round(60.0 / p)
    if abs(snap / p - 1) < 0.002:
        sc, d = max((_score(env, t0 + d, snap), d) for d in np.arange(-0.006, 0.0061, 0.001))
        if sc >= 0.99 * best[0]:
            t0, p = t0 + d, snap

    offsets = []
    for q in range(8):
        lo, hi = dur * q / 8, dur * (q + 1) / 8
        ds = np.arange(-0.1, 0.1001, 0.004)
        sc = np.array([_score(env, t0 + d, p, lo, hi) for d in ds])
        if sc.max() > 3 * max(sc.mean(), 1e-9):
            offsets.append(float(ds[int(np.argmax(sc))]))
    stable = bool(dur > 20 and len(offsets) >= 3 and max(abs(o) for o in offsets) < 0.03)
    return {"bpm": 60.0 / p, "t0": t0, "period": p, "env": env, "grid_stable": stable, "offsets": offsets}


def grid_strength(g: dict, dur: float) -> float:
    env = g["env"]
    at = np.interp((g["t0"] + np.arange(int((dur - g["t0"]) / g["period"])) * g["period"]) * ASR / HOP, np.arange(len(env)), env)
    return float(at.mean() / max(env.mean(), 1e-9))


def pick_grid(mix: dict, drums: np.ndarray | None, dur: float) -> tuple[dict, str]:
    if drums is None:
        return mix, "mix"
    gd = beat_grid(drums)
    if abs(gd["bpm"] / mix["bpm"] - 1) < 0.005 or grid_strength(gd, dur) >= 1.5:
        return gd, "drums_stem"
    return mix, "mix"


def downbeat_estimate(ys: list[np.ndarray], t0: float, p: float) -> tuple[float, dict]:
    n = max(int((len(ys[0]) / ASR - t0) / p), 4)
    ks = np.arange(n)
    means = np.zeros(4)
    for y in ys:
        kick = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, fmax=250, n_mels=32)
        vals = np.interp((t0 + ks * p) * ASR / HOP, np.arange(len(kick)), kick) / max(float(kick.max()), 1e-9)
        means += np.array([vals[ph::4].mean() for ph in range(4)])
    phase = int(np.argmax(means))
    best, second = np.sort(means)[::-1][:2]
    contrast = float((best - second) / max(best, 1e-9))
    return max(t0 + phase * p - ONSET_LATENCY, 0.0), {"phase_contrast": contrast}
