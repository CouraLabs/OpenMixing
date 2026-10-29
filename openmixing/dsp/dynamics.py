from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..core.units import SR

RATE = 1000
BLK = SR // RATE
HOLD = 5


def _level(x: np.ndarray) -> np.ndarray:
    a = np.abs(np.atleast_2d(x)).max(axis=0)
    n = -(-len(a) // BLK)
    a = np.pad(a, (0, n * BLK - len(a))).reshape(n, BLK).max(axis=1)
    return maximum_filter1d(a, HOLD)


def _follow(v: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    ca, cr = np.exp(-1.0 / max(attack_ms * RATE / 1000, 1e-3)), np.exp(-1.0 / max(release_ms * RATE / 1000, 1e-3))
    out, s = np.empty(len(v)), 0.0
    for i, t in enumerate(v.tolist()):
        c = ca if t > s else cr
        s = c * s + (1 - c) * t
        out[i] = s
    return out


def _up(v: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.arange(n) / BLK - 0.5, np.arange(len(v)), v).astype(np.float32)


def envelope(x: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    n = np.atleast_2d(x).shape[1]
    return _up(_follow(_level(x), attack_ms, release_ms), n)


def _db(v: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(v, 1e-6))


def static_reduction(lvl_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    over = lvl_db - threshold_db
    k = max(knee_db, 1e-6)
    soft = (1 - 1 / ratio) * (over + k / 2) ** 2 / (2 * k)
    return np.where(2 * over < -k, 0.0, np.where(2 * over <= k, soft, (1 - 1 / ratio) * over))


def _apply(x: np.ndarray, gr_db: np.ndarray, attack_ms: float, release_ms: float, makeup_db: float = 0.0) -> np.ndarray:
    g = _up(_follow(gr_db, attack_ms, release_ms), x.shape[1])
    return (x * np.float32(10 ** (makeup_db / 20)) * np.power(np.float32(10), -g / 20)).astype(np.float32)


def compress(x: np.ndarray, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, knee_db: float = 6.0,
             makeup_db: float = 0.0, key: np.ndarray | None = None) -> np.ndarray:
    lvl = _db(_level(x if key is None else key))
    return _apply(x, static_reduction(lvl, threshold_db, ratio, knee_db), attack_ms, release_ms, makeup_db)


def duck(x: np.ndarray, key: np.ndarray, depth_db: float, attack_ms: float, release_ms: float, threshold_db: float) -> np.ndarray:
    gr = np.where(_db(_level(key)) >= threshold_db, depth_db, 0.0)
    return _apply(x, gr, attack_ms, release_ms)


def pump(x: np.ndarray, beat_sec: float, depth_db: float, release_beats: float = 0.8, beats: float = 1.0) -> np.ndarray:
    period = max(beat_sec * beats, 1e-3)
    t = (np.arange(x.shape[1]) / SR % period) / beat_sec
    gr = depth_db * np.exp(-3.0 * t / max(release_beats, 1e-3))
    return (x * np.power(np.float32(10), -gr.astype(np.float32) / 20)).astype(np.float32)
