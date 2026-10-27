"""Synthetic tracks with a known beat grid, for demos and tests."""
from __future__ import annotations

import numpy as np

from .models import SR, Track


def make_track(id: str, bpm: float, bars: int, first_downbeat: float = 0.0, click_hz: float = 1000.0,
               bass_hz: float = 55.0, pad_hz: float | None = None) -> Track:
    """Click on every beat (louder on the downbeat) + a bass note per beat + optional pad."""
    beat = 60.0 / bpm
    n = int((first_downbeat + bars * 4 * beat) * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float64)
    for i in range(bars * 4):
        s = int(round((first_downbeat + i * beat) * SR))
        m = min(int(0.08 * SR), n - s)
        if m <= 0:
            break
        tt = np.arange(m) / SR
        amp = 0.8 if i % 4 == 0 else 0.5
        env = np.exp(-tt * 60)
        x[s : s + m] += amp * env * np.sin(2 * np.pi * click_hz * tt)
        x[s : s + m] += 0.5 * env * np.sin(2 * np.pi * bass_hz * tt)
    if pad_hz:
        x += 0.1 * np.sin(2 * np.pi * pad_hz * t)
    audio = np.stack([x, x]).astype(np.float32)
    return Track(id, audio, bpm, first_downbeat)
