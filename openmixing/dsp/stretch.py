from __future__ import annotations

import numpy as np
from pedalboard import time_stretch

from ..core.units import SR

CONTEXT = 4096
XFADE = 480
STEP_PROBE = 256
MAX_PITCH_STEP = 0.25


def read(audio: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = np.zeros((audio.shape[0], max(stop - start, 0)), dtype=np.float32)
    lo, hi = max(start, 0), min(stop, audio.shape[1])
    if hi > lo:
        out[:, lo - start : hi - start] = audio[:, lo:hi]
    return out


def fit(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def _stretch(x: np.ndarray, ratio: float, semitones: float) -> np.ndarray:
    return time_stretch(np.ascontiguousarray(x, dtype=np.float32), SR, ratio, pitch_shift_in_semitones=semitones,
                        high_quality=True)


def pitch_chunk(semi: np.ndarray, chunk: int) -> int:
    if len(semi) < 2 * STEP_PROBE:
        return chunk
    slope = float(np.max(np.abs(np.diff(semi[::STEP_PROBE])))) / STEP_PROBE
    return chunk if slope <= 0 else int(np.clip(MAX_PITCH_STEP / slope, 4096, chunk))


def play(audio: np.ndarray, src_start: float, speed, n_out: int, semitones=0.0,
         key_lock: bool = True, chunk: int = 48_000) -> np.ndarray:
    if n_out <= 0:
        return np.zeros((audio.shape[0], 0), np.float32)
    spd = np.full(n_out, float(speed)) if np.isscalar(speed) else np.asarray(speed, dtype=np.float64)
    start = int(round(src_start))
    semi = np.full(n_out, float(semitones)) if np.isscalar(semitones) else np.array(semitones, dtype=np.float64)
    if np.all(np.abs(spd - 1) < 1e-9) and np.all(np.abs(semi) < 1e-9):
        return read(audio, start, start + n_out)
    chunk = pitch_chunk(semi, max(int(chunk), 256))
    for o0 in range(0, n_out, chunk):
        spd[o0 : o0 + chunk] = spd[o0 : o0 + chunk].mean()
        semi[o0 : o0 + chunk] = semi[o0 : o0 + chunk].mean()
    pos = np.concatenate([[0.0], np.cumsum(spd)])
    out = np.zeros((audio.shape[0], n_out), dtype=np.float32)
    for o0 in range(0, n_out, chunk):
        o1 = min(o0 + chunk, n_out)
        e0, e1 = max(o0 - XFADE, 0), min(o1 + XFADE, n_out)
        s0, s1 = int(round(pos[e0])), int(round(pos[e1]))
        src = read(audio, start + s0 - CONTEXT, start + s1 + CONTEXT)
        ratio = float(spd[o0])
        st = float(semi[o0]) + (0.0 if key_lock else 12 * np.log2(ratio))
        if abs(ratio - 1) < 1e-9 and abs(st) < 1e-9:
            piece = src[:, CONTEXT : CONTEXT + e1 - e0]
        else:
            y = _stretch(src, ratio, st)
            piece = y[:, int(round(CONTEXT / ratio)) :]
        piece = fit(piece, e1 - e0)
        w = np.ones(e1 - e0, dtype=np.float32)
        n = min(2 * XFADE, len(w))
        if e0 > 0:
            w[:n] = np.linspace(0, 1, n, endpoint=False)
        if e1 < n_out:
            w[len(w) - n :] *= np.linspace(1, 0, n, endpoint=False)
        out[:, e0:e1] += piece * w
    return out
