from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

BLOCK = 48
SMOOTH = 40


def limit(x: np.ndarray, ceiling_db: float = -1.5) -> np.ndarray:
    ceil = 10 ** (ceiling_db / 20)
    n = x.shape[1]
    nb = -(-n // BLOCK)
    peak = np.abs(np.pad(x, ((0, 0), (0, nb * BLOCK - n)))).max(0).reshape(nb, BLOCK).max(1)
    g = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    if g.min() >= 1.0:
        return x
    g = minimum_filter1d(g, 2 * SMOOTH - 1, mode="nearest")
    g = uniform_filter1d(g, SMOOTH, mode="nearest")
    gs = np.interp(np.arange(n), (np.arange(nb) + 0.5) * BLOCK, g).astype(np.float32)
    return x * gs
