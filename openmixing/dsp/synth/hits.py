from __future__ import annotations

import numpy as np

from ...core.units import SR
from .noise import highpass, noise, unit_rms

SWEEP_TAU_S = 0.08


def pan_gain(y: np.ndarray, pan: float) -> np.ndarray:
    return (y * np.array([[min(1.0, 1.0 - pan)], [min(1.0, 1.0 + pan)]], np.float32)).astype(np.float32)


def decay(n: int, decay_s: float) -> np.ndarray:
    return np.exp(-6.9078 * np.arange(n) / SR / decay_s)


def sine_sweep(n: int, f_start: float, f_end: float) -> np.ndarray:
    f = f_end + (f_start - f_end) * np.exp(-np.arange(n) / SR / SWEEP_TAU_S)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def fade_out(y: np.ndarray) -> np.ndarray:
    k = min(int(0.005 * SR), y.shape[1] // 2)
    if k:
        y[:, -k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return y


def impact(n: int, rng: np.random.Generator, f_start: float, f_end: float, decay_s: float, noise_mix: float, pan: float) -> np.ndarray:
    body = sine_sweep(n, f_start, f_end)
    burst = unit_rms(highpass(noise(n, 0, rng), 200.0)).astype(np.float64)
    y = (body[None] + noise_mix * burst * np.exp(-np.arange(n) / SR / 0.15)) * decay(n, decay_s)
    return fade_out(pan_gain(unit_rms(y), pan))


def crash(n: int, rng: np.random.Generator, hp_hz: float, decay_s: float, pan: float) -> np.ndarray:
    return fade_out(pan_gain(unit_rms(highpass(noise(n, 0, rng), hp_hz) * decay(n, decay_s)), pan))


def snare(n: int, rng: np.random.Generator, pitch_hz: float, decay_s: float, pan: float) -> np.ndarray:
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * pitch_hz * t) * np.exp(-t / 0.05)
    burst = highpass(noise(n, 0, rng), 1500.0) * decay(n, decay_s)
    return fade_out(pan_gain(unit_rms(0.6 * body[None] + 0.8 * unit_rms(burst)), pan))


def flat_noise(n: int, rng: np.random.Generator, color: int, pan: float) -> np.ndarray:
    return fade_out(pan_gain(noise(n, color, rng), pan))
