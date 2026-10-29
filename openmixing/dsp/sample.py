from __future__ import annotations

import numpy as np

from ..core.units import SR
from . import stretch, synth

FADE = SR // 1000
CHOKE_FADE = SR // 200
FITS = ("none", "stretch_to_beats", "loop_for_duration")
PARAMS = {"pitch_semitones": (0.0, -24.0, 24.0), "fit": (0.0, 0.0, 2.0), "velocity": (0.0, -24.0, 12.0), "choke": (0.0, 0.0, 1.0)}
KINDS = [*synth.KINDS, "sample"]


def param_spec(kind: str) -> dict:
    return PARAMS if kind == "sample" else synth.KINDS[kind]["params"]


def voice(audio: np.ndarray, kind: str, root_bpm: float | None, bpm: float, n: int, semitones: float, fit: int) -> np.ndarray:
    length = audio.shape[1]
    if fit == 1:
        y = stretch.play(audio, 0, length / n, n, semitones)
    else:
        speed = bpm / root_bpm if kind == "loop" and root_bpm else 1.0
        src = np.tile(audio, (1, int(np.ceil(n * speed / length)) + 1)) if fit == 2 else audio
        y = stretch.play(src, 0, speed, n, semitones)
    y = y / max(float(np.abs(audio).max()), 1e-9)
    f = min(FADE, n // 2)
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    y[:, :f] *= ramp
    y[:, n - f :] *= ramp[::-1]
    return y.astype(np.float32)


def cut(y: np.ndarray, k: int) -> None:
    k = max(min(k, y.shape[1]), 0)
    f = min(CHOKE_FADE, k)
    y[:, k:] = 0.0
    if f:
        y[:, k - f : k] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
