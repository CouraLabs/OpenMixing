from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from ..core.units import SR
from .stretch import read

BLOCK = SR
ALIAS_SPEED = 1.5
STOP_SPEED = 0.25
STOP_POWER = 0.3


def hermite(win: np.ndarray, i: np.ndarray, t: np.ndarray) -> np.ndarray:
    y0, y1, y2, y3 = (win[:, i + k] for k in range(4))
    c1, c2, c3 = 0.5 * (y2 - y0), y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3, 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
    return ((c3 * t + c2) * t + c1) * t + y1


def vinyl(audio: np.ndarray, positions: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    out = np.zeros((audio.shape[0], len(positions)), np.float32)
    top = float(np.abs(speeds).max()) if len(speeds) else 0.0
    sos = butter(4, 0.9 / top, output="sos") if top > ALIAS_SPEED else None
    for o in range(0, len(positions), BLOCK):
        pos = positions[o : o + BLOCK]
        i = np.floor(pos).astype(np.int64)
        lo = int(i.min()) - 1
        win = read(audio, lo, int(i.max()) + 3).astype(np.float64)
        if sos is not None:
            win = sosfilt(sos, win, axis=1)
        out[:, o : o + BLOCK] = hermite(win, i - lo - 1, pos - i)
    return out * np.minimum(1.0, np.abs(speeds) / STOP_SPEED).astype(np.float32) ** STOP_POWER
