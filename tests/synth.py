"""Synthetic club-style tracks for tests: kick on every beat, hat off-beats, bass note, pad."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from openmixing.models import SR


def make_track(path, bpm=120.0, bars=32, bass_hz=55.0, pad_hz=330.0, lead_delay=0.0, sr=SR):
    beat = 60.0 / bpm
    n = int((bars * 4 * beat + lead_delay) * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(int(bars * 4)):
        i = int((lead_delay + k * beat) * sr)
        m = min(int(0.25 * sr), n - i)
        tt = np.arange(m) / sr
        y[i : i + m] += np.sin(2 * np.pi * (45 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12) * 0.9
        j = i + int(beat / 2 * sr)
        m = min(int(0.04 * sr), n - j)
        if m > 0:
            y[j : j + m] += np.random.default_rng(k).standard_normal(m) * np.exp(-np.arange(m) / sr * 150) * 0.08
    y += 0.12 * np.sin(2 * np.pi * bass_hz * t) + 0.08 * np.sin(2 * np.pi * pad_hz * t)
    y = y / max(abs(y).max(), 1) * 0.8
    sf.write(path, np.stack([y, y], 1).astype(np.float32), sr, subtype="PCM_24")
    return path
