from __future__ import annotations

import numpy as np
from pedalboard import Distortion

from ...core.units import SR
from .catalog import FxOut


def brake(region, beat_sec, beats, p, name):
    L = region.shape[1]
    dur = int(min(beats * beat_sec * SR, L))
    t = np.arange(dur, dtype=np.float64)
    c = float(p["curve"])
    u = t / dur
    pos = dur * (1 - (1 - u) ** (1 + c)) / (1 + c)
    out = np.zeros_like(region)
    for ch in range(region.shape[0]):
        out[ch, :dur] = np.interp(pos, np.arange(L), region[ch])
    out[:, :dur] *= np.linspace(1, 0.0, dur, dtype=np.float32) ** 0.3
    return FxOut(out)


def _slices(region, beat_sec, beats, repeat: bool):
    L = region.shape[1]
    n = max(int(beats * beat_sec * SR), 16)
    out = np.empty_like(region)
    f = min(int(0.002 * SR), n // 4)
    fade = np.ones(n, np.float32)
    if f:
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
    if repeat:
        base = region[:, :n] if region.shape[1] >= n else np.pad(region, ((0, 0), (0, n - L)))
        base = base * fade
        reps = -(-L // n)
        return np.tile(base, (1, reps))[:, :L]
    for i in range(0, L, n):
        seg = region[:, i : i + n][:, ::-1]
        out[:, i : i + seg.shape[1]] = seg * fade[: seg.shape[1]] if seg.shape[1] == n else seg
    return out


def repeater(region, beat_sec, beats, p, name):
    return FxOut(_slices(region, beat_sec, beats, repeat=True))


def reverse(region, beat_sec, beats, p, name):
    return FxOut(_slices(region, beat_sec, beats, repeat=False))


def bitcrush(region, beat_sec, beats, p, name):
    levels = 2 ** (int(p["bits"]) - 1)
    y = np.round(region * levels) / levels
    k = max(int(p["downsample"]), 1)
    if k > 1:
        y = np.repeat(y[:, ::k], k, axis=1)[:, :region.shape[1]]
    return FxOut(y.astype(np.float32))


def distortion(region, beat_sec, beats, p, name):
    return FxOut(Distortion(drive_db=float(p["drive_db"]))(region, SR))
