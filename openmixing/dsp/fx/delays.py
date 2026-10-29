from __future__ import annotations

import numpy as np
from pedalboard import Reverb
from scipy.signal import lfilter

from ...core.units import SR
from .catalog import MAX_TAIL_S, FxOut


def _tail_for(beat_sec: float, beats: float, feedback: float | None = None) -> int:
    t = beats * beat_sec
    if feedback is not None:
        reps = np.log(1e-3) / np.log(max(min(feedback, 0.95), 0.05))
        t *= reps
    return int(min(t, MAX_TAIL_S) * SR)


def _feedback_delay(x: np.ndarray, delay_n: int, feedback: float, damping_hz: float | None,
                    pingpong: bool = False) -> np.ndarray:
    n = x.shape[1]
    D = max(int(delay_n), 1)
    src = x.mean(0, keepdims=True) if pingpong else x
    out = np.zeros_like(x)
    if damping_hz:
        a = float(np.exp(-2 * np.pi * damping_hz / SR))
    zi = np.zeros((src.shape[0], 1))
    prev = None
    for k in range(-(-n // D)):
        i, j = k * D, min(k * D + D, n)
        if prev is None:
            wet = np.zeros((src.shape[0], j - i), np.float32)
        else:
            wet = prev[:, : j - i]
            if damping_hz:
                wet, zi = lfilter([1 - a], [1, -a], wet, axis=-1, zi=zi)
            wet = (wet * feedback).astype(np.float32)
            if pingpong:
                out[k % 2, i:j] = wet[0]
            else:
                out[:, i:j] = wet
        prev = np.zeros((src.shape[0], D), np.float32)
        prev[:, : j - i] = src[:, i:j] + wet
    return out


def echo(region, beat_sec, beats, p, name):
    D = int(beats * beat_sec * SR)
    fb = float(p["feedback"])
    tail = _tail_for(beat_sec, beats, fb)
    x = np.pad(region, ((0, 0), (0, tail)))
    wet = _feedback_delay(x, D, fb, p.get("damping_hz"), pingpong=(name == "ping_pong"))
    return FxOut(wet / max(fb, 1e-3), dry_cut=(name == "echo_out"))


def reverb(region, beat_sec, beats, p, name):
    decay = beats * beat_sec
    room = float(np.clip(decay / 8.0, 0.2, 1.0)) if name == "reverb" else 0.97
    tail = int(min(max(decay, 1.0) * (1.5 if name == "reverb" else 2.5), MAX_TAIL_S) * SR)
    x = np.pad(region, ((0, 0), (0, tail)))
    rv = Reverb(room_size=room, damping=float(p["damping"]), wet_level=1.0, dry_level=0.0, width=1.0)
    return FxOut(rv(x, SR))
