from __future__ import annotations

import numpy as np

from ..core.units import SR
from ..dsp.vinyl import vinyl
from ..model.events import RateEvent
from ..model.rate import speed_at
from ..model.tempo import TempoMap


def play_rate(audio: np.ndarray, ev: RateEvent, s0: float, tm: TempoMap, src_bpm: float, fd: float,
              seg: tuple[int, int], win: tuple[int, int]) -> np.ndarray:
    (n0, n1), (a, b) = seg, win
    ob = tm.sec_to_beat(np.arange(n0, b) / SR)
    spd = speed_at(ev, (ob - ev.beat) / ev.duration_beats)
    inc = 0.5 * (spd[1:] + spd[:-1]) * np.diff(ob)
    src_beat = s0 + np.concatenate([[0.0], np.cumsum(inc)])
    pos = (fd + src_beat * 60.0 / src_bpm) * SR
    y = vinyl(audio, pos[a - n0 :], spd[a - n0 :])
    f = min(int(0.001 * SR), y.shape[1] // 4)
    if f and a == n0:
        y[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
    if f and b == n1:
        y[:, -f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return y
