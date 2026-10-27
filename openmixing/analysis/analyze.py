"""Beat-grid and key primitives (constant-tempo grid fit + downbeat), used by features.py.

Uses librosa for beat tracking (the plan's documented fallback to beat_this) and refines the grid
against the onset envelope so beatmatching between tracks is accurate to a few ms.
"""
from __future__ import annotations

import librosa
import numpy as np
import soxr

from ..models import SR

AUDIO_EXT = {".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".aac", ".ogg", ".opus", ".alac"}
ASR, HOP = 22050, 256
TARGET_LUFS = -14.0
ONSET_LATENCY = 0.013  # s; measured bias of the onset envelope vs true transient position


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
    """Sum of the (detrended) onset envelope on the grid t0 + k*period within [lo, hi) seconds."""
    hi = len(env) * HOP / ASR if hi is None else hi
    k = np.arange(int(np.ceil((lo - t0) / period)), int(np.ceil((hi - t0) / period)))
    t = (t0 + k * period) * ASR / HOP
    t = t[(t >= 0) & (t < len(env) - 1)]
    return float(np.interp(t, np.arange(len(env)), env).sum()) if len(t) else -1e9


def beat_grid(y: np.ndarray) -> dict:
    """Constant-tempo grid: librosa tempo as a seed, then a global fit of tempo + phase to the onset envelope.

    librosa's individual beat positions wander (phase slips, half-beat jumps), so only its tempo is
    trusted; the phase and the exact period come from maximising onset energy on a rigid grid.
    Stability = the best phase of each quarter of the track agrees with the global one.
    """
    env = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP)
    env = np.maximum(env - np.convolve(env, np.ones(43) / 43, mode="same"), 0)  # drop slow trend
    dur = len(y) / ASR
    seed = float(np.atleast_1d(librosa.beat.beat_track(onset_envelope=env, sr=ASR, hop_length=HOP, start_bpm=124)[0])[0])
    p0 = 60.0 / _fold(seed)

    best = (-1e9, 0.0, p0)
    for pf in np.linspace(0.985, 1.015, 61):  # coarse: period ±1.5%, phase over a full beat
        p = p0 * pf
        for t0 in np.arange(0, p, 0.006):
            sc = _score(env, t0, p)
            if sc > best[0]:
                best = (sc, t0, p)
    _, t0, p = best
    for pf in np.linspace(0.9985, 1.0015, 31):  # fine
        for d in np.arange(-0.008, 0.0081, 0.001):
            sc = _score(env, t0 + d, p * pf)
            if sc > best[0]:
                best = (sc, t0 + d, p * pf)
    _, t0, p = best

    # Per-eighth phase check. Only segments with a clear beat (sharp score peak) count as evidence.
    offsets = []
    for q in range(8):
        lo, hi = dur * q / 8, dur * (q + 1) / 8
        ds = np.arange(-0.1, 0.1001, 0.004)
        sc = np.array([_score(env, t0 + d, p, lo, hi) for d in ds])
        if sc.max() > 3 * max(sc.mean(), 1e-9):
            offsets.append(float(ds[int(np.argmax(sc))]))
    stable = bool(dur > 20 and len(offsets) >= 3 and max(abs(o) for o in offsets) < 0.03)
    return {"bpm": 60.0 / p, "t0": t0, "period": p, "env": env, "grid_stable": stable, "offsets": offsets}


def downbeat_estimate(y: np.ndarray, t0: float, p: float) -> tuple[float, dict]:
    """First downbeat: the beat phase (mod 4) with the strongest low-frequency (kick) onsets, and how clearly it wins."""
    kick = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, fmax=250, n_mels=32)
    ks = np.arange(max(int((len(y) / ASR - t0) / p), 4))
    vals = np.interp((t0 + ks * p) * ASR / HOP, np.arange(len(kick)), kick)
    means = np.array([vals[ph::4].mean() for ph in range(4)])
    phase = int(np.argmax(means))
    best, second = np.sort(means)[::-1][:2]
    contrast = float((best - second) / max(best, 1e-9))
    return max(t0 + phase * p - ONSET_LATENCY, 0.0), {"phase_contrast": contrast}


def downbeat(y: np.ndarray, t0: float, p: float) -> float:
    return downbeat_estimate(y, t0, p)[0]
