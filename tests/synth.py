"""Synthetic club-style tracks for tests: kick on every beat, hat off-beats, bass note, pad."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from openmixing.core.units import SR


def make_track(path, bpm=120.0, bars=32, bass_hz=55.0, pad_hz=330.0, lead_delay=0.0, sr=SR, vocal_bars=(), riser_bars=None):
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
    bar_n = int(4 * beat * sr)
    for k in vocal_bars:
        a = int(lead_delay * sr) + k * bar_n
        tt = t[a : a + bar_n]
        y[a : a + bar_n] += sum(0.12 * np.sin(2 * np.pi * f * tt) for f in (220, 440, 660, 880))
    if riser_bars:
        r0, r1 = riser_bars
        for k in range(r0, r1):
            a = int(lead_delay * sr) + k * bar_n
            frac = (k - r0) / max(r1 - r0 - 1, 1)
            sos = butter(4, 500 + 2500 * frac, "highpass", fs=sr, output="sos")
            seg = sosfilt(sos, np.random.default_rng(k).standard_normal(len(y[a : a + bar_n])))
            y[a : a + bar_n] += seg * (0.05 + 0.4 * frac)
    y = y / max(abs(y).max(), 1) * 0.8
    sf.write(path, np.stack([y, y], 1).astype(np.float32), sr, subtype="PCM_24")
    return path
