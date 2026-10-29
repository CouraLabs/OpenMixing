from __future__ import annotations

import numpy as np
from pedalboard import Chorus, LadderFilter, Phaser

from ...core.units import SR
from .catalog import FxOut


def _rate(beat_sec, beats, lo=0.05, hi=20.0):
    return float(np.clip(1.0 / (beats * beat_sec), lo, hi))


def lfo_filter(region, beat_sec, beats, p, name):
    out = np.empty_like(region)
    f = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=float(p["max_hz"]), resonance=float(p["resonance"]))
    lo, hi = float(p["min_hz"]), float(p["max_hz"])
    period = beats * beat_sec * SR
    B = 128
    for i in range(0, region.shape[1], B):
        ph = 0.5 - 0.5 * np.cos(2 * np.pi * (i + B / 2) / period)
        f.cutoff_hz = float(hi * (lo / hi) ** ph)
        out[:, i : i + B] = f(region[:, i : i + B], SR, reset=False)
    return FxOut(out)


def flanger(region, beat_sec, beats, p, name):
    fx = Chorus(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_delay_ms=2.0, feedback=float(p["feedback"]), mix=0.6)
    return FxOut(fx(region, SR))


def phaser(region, beat_sec, beats, p, name):
    fx = Phaser(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_frequency_hz=1300.0, feedback=float(p["feedback"]), mix=0.7)
    return FxOut(fx(region, SR))


def chorus(region, beat_sec, beats, p, name):
    fx = Chorus(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_delay_ms=7.0, feedback=0.0, mix=0.5)
    return FxOut(fx(region, SR))


def tremolo(region, beat_sec, beats, p, name):
    t = np.arange(region.shape[1]) / SR
    g = 1 - float(p["depth"]) * (0.5 - 0.5 * np.cos(2 * np.pi * t / (beats * beat_sec)))
    return FxOut(region * g.astype(np.float32))


def transform(region, beat_sec, beats, p, name):
    ph = (np.arange(region.shape[1]) / SR / (beats * beat_sec)) % 1.0
    k = max(int(0.002 * SR), 1)
    gate = np.convolve((ph < float(p["duty"])).astype(np.float32), np.ones(k, np.float32) / k, mode="same")
    return FxOut(region * gate)
