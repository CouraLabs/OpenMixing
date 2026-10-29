from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt

from ...core.units import SR
from .noise import noise, unit_rms

BLOCK = 512
FADE = int(0.002 * SR)


def bandpass_sos(fc: float, q: float = 4.0) -> np.ndarray:
    w = 2 * np.pi * fc / SR
    alpha = np.sin(w) / (2 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w), 1 - alpha])
    return np.concatenate([b, a])[None] / (1 + alpha)


def sweep(n: int, f0: float, f1: float) -> np.ndarray:
    return f0 * (f1 / f0) ** (np.arange(n) / max(n - 1, 1))


def filtered_noise(n: int, f: np.ndarray, color: int, rng: np.random.Generator) -> np.ndarray:
    x = noise(n, color, rng).astype(np.float64)
    out = np.zeros_like(x)
    zi = np.zeros((1, 2, 2))
    for i in range(0, n, BLOCK):
        fc = float(np.clip(f[min(i + BLOCK // 2, n - 1)], 30.0, 0.45 * SR))
        out[:, i : i + BLOCK], zi = sosfilt(bandpass_sos(fc), x[:, i : i + BLOCK], axis=-1, zi=zi)
    return out


def tonal(f: np.ndarray) -> np.ndarray:
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.5 * np.sin(2 * ph) * (2 * f < 0.45 * SR)
    return np.stack([y, y])


def riser(n: int, rng: np.random.Generator, f0: float, f1: float, tone: float, color: int, widen: float, pan: float) -> np.ndarray:
    if n < 2:
        return np.zeros((2, max(n, 0)), np.float32)
    f = sweep(n, f0, f1)
    u = np.linspace(0.0, 1.0, n)
    y = (1 - tone) * unit_rms(filtered_noise(n, f, color, rng))
    if tone > 0:
        y = y + tone * unit_rms(tonal(f))
    w = widen * u
    y[1] = y[0] * (1 - w) + y[1] * w
    y = unit_rms(y * (u ** 2 + 1e-3))
    y *= np.array([[min(1.0, 1.0 - pan)], [min(1.0, 1.0 + pan)]])
    k = min(FADE, n // 2)
    y[:, :k] *= np.linspace(0, 1, k)
    return y.astype(np.float32)


def downlifter(n: int, rng: np.random.Generator, f0: float, f1: float, tone: float, color: int, widen: float, pan: float) -> np.ndarray:
    return np.ascontiguousarray(riser(n, rng, f0, f1, tone, color, widen, pan)[:, ::-1])
