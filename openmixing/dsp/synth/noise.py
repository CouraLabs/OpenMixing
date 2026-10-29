from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from ...core.units import SR

PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]
COLORS = ("white", "pink", "brown")


def unit_rms(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.sqrt((x.astype(np.float64) ** 2).mean())), 1e-9)


def noise(n: int, color: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((2, n))
    if int(color) == 1:
        x = lfilter(PINK_B, PINK_A, x, axis=-1)
    elif int(color) == 2:
        x = lfilter([1.0], [1.0, -0.995], x, axis=-1)
    return unit_rms(x).astype(np.float32)


def highpass(x: np.ndarray, hz: float) -> np.ndarray:
    return sosfilt(butter(2, hz, "highpass", fs=SR, output="sos"), x, axis=-1).astype(np.float32)
