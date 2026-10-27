"""Three-band DJ EQ (low / mid / high) with kill, plus the resonant DJ 'filter' knob."""
from __future__ import annotations

import numpy as np
from pedalboard import LadderFilter
from scipy.signal import butter, sosfilt

from ..models import SR

KILL_DB = -80.0
LOW_XOVER, HIGH_XOVER = 200.0, 2500.0  # Hz
MAX_BOOST_DB = 6.0  # Serato-style knob: kill ... +6 dB


def db_to_gain(db: np.ndarray | float):
    return np.where(np.asarray(db) <= KILL_DB + 1e-6, 0.0, 10 ** (np.asarray(db) / 20))


def _lr4(x: np.ndarray, fc: float, kind: str) -> np.ndarray:
    sos = butter(2, fc, btype=kind, fs=SR, output="sos")
    return sosfilt(np.vstack([sos, sos]), x, axis=-1).astype(np.float32)  # two cascaded = Linkwitz-Riley 4


def split_bands(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low / mid / high from true LR4 crossovers. The low band is run through the 2.5 kHz all-pass
    (LP + HP) so the bands sum back to |x| with flat magnitude; a killed band leaves < -40 dB behind."""
    low = _lr4(x, LOW_XOVER, "low")
    rest = _lr4(x, LOW_XOVER, "high")
    low = _lr4(low, HIGH_XOVER, "low") + _lr4(low, HIGH_XOVER, "high")
    return low, _lr4(rest, HIGH_XOVER, "low"), _lr4(rest, HIGH_XOVER, "high")


def apply_eq(x: np.ndarray, low_db: np.ndarray | None, mid_db: np.ndarray | None, high_db: np.ndarray | None) -> np.ndarray:
    """Per-sample dB envelopes (None = untouched band). Returns x unchanged when all are None / neutral."""
    envs = [e for e in (low_db, mid_db, high_db)]
    if all(e is None or np.all(np.abs(e) < 1e-6) for e in envs):
        return x
    low, mid, high = split_bands(x)
    g = [np.ones(x.shape[1], np.float32) if e is None else db_to_gain(e).astype(np.float32) for e in envs]
    return low * g[0] + mid * g[1] + high * g[2]


def apply_filter_knob(x: np.ndarray, value: np.ndarray, block: int = 256) -> np.ndarray:
    """DJ filter knob. value in [-1, 1]: <0 low-pass sweeping 20 kHz -> ~80 Hz, >0 high-pass 20 Hz -> ~9 kHz."""
    if np.all(np.abs(value) < 1e-3):
        return x
    out = x.copy()
    lp = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=20000, resonance=0.25)
    hp = LadderFilter(mode=LadderFilter.Mode.HPF24, cutoff_hz=20, resonance=0.25)
    active = np.flatnonzero(np.abs(value[::block]) >= 1e-3)
    if not len(active):
        return x
    for bi in range(active[0], active[-1] + 1):
        i, j = bi * block, min(bi * block + block, x.shape[1])
        v = float(value[min(i + block // 2, x.shape[1] - 1)])
        if abs(v) < 1e-3:
            lp.reset(), hp.reset()
            continue
        if v < 0:
            lp.cutoff_hz = float(20000 * (80 / 20000) ** min(-v, 1.0))
            out[:, i:j] = lp(x[:, i:j], SR, reset=False)
        else:
            hp.cutoff_hz = float(20 * (9000 / 20) ** min(v, 1.0))
            out[:, i:j] = hp(x[:, i:j], SR, reset=False)
    return out
