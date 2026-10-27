"""Beat-accurate tempo change: Rubber Band in short chunks, each placed at its exact output position.

pedalboard's time_stretch drifts ~0.1% over long audio and mishandles array ratios, so playback is cut
into short chunks (default 2 beats), each stretched at a constant ratio with source context on both sides
and cross-faded into place. Beat error stays ~1 ms and a tempo ramp becomes a fine staircase.
"""
from __future__ import annotations

import numpy as np
from pedalboard import time_stretch

from ..models import SR

CONTEXT = 4096  # source samples of context around each chunk
XFADE = 480     # half-length (samples) of the cross-fade between chunks


def read(audio: np.ndarray, start: int, stop: int) -> np.ndarray:
    """audio[:, start:stop] with out-of-range parts as silence."""
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


def play(audio: np.ndarray, src_start: float, speed, n_out: int, semitones: float = 0.0,
         key_lock: bool = True, chunk: int = 48_000) -> np.ndarray:
    """`n_out` output samples of `audio` starting at source sample `src_start`.

    speed: scalar or per-output-sample array (source samples advanced per output sample).
    key_lock False = varispeed (pitch follows tempo); semitones is an additional transposition.
    """
    if n_out <= 0:
        return np.zeros((audio.shape[0], 0), np.float32)
    spd = np.full(n_out, float(speed)) if np.isscalar(speed) else np.asarray(speed, dtype=np.float64)
    start = int(round(src_start))
    if np.all(np.abs(spd - 1) < 1e-9) and semitones == 0:
        return read(audio, start, start + n_out)
    chunk = max(int(chunk), 256)
    for o0 in range(0, n_out, chunk):
        spd[o0 : o0 + chunk] = spd[o0 : o0 + chunk].mean()
    pos = np.concatenate([[0.0], np.cumsum(spd)])
    out = np.zeros((audio.shape[0], n_out), dtype=np.float32)
    for o0 in range(0, n_out, chunk):
        o1 = min(o0 + chunk, n_out)
        e0, e1 = max(o0 - XFADE, 0), min(o1 + XFADE, n_out)
        s0, s1 = int(round(pos[e0])), int(round(pos[e1]))
        src = read(audio, start + s0 - CONTEXT, start + s1 + CONTEXT)
        ratio = float(spd[o0])
        semi = semitones + (0.0 if key_lock else 12 * np.log2(ratio))
        if abs(ratio - 1) < 1e-9 and abs(semi) < 1e-9:
            piece = src[:, CONTEXT : CONTEXT + e1 - e0]
        else:
            y = _stretch(src, ratio, semi)
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
