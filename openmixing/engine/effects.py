from __future__ import annotations

import numpy as np

from ..core.units import SR
from ..dsp import fx as fxm
from ..model.clip import Clip
from ..model.tempo import TempoMap


def wet_envelope(ev, tm: TempoMap, a: int, b: int, na: int) -> float | np.ndarray:
    if not ev.wet_curve:
        return ev.wet
    pts = np.asarray(ev.wet_curve, np.float64)
    beats = tm.sec_to_beat(np.arange(a + na, b + na) / SR) - ev.beat
    return (ev.wet * np.interp(beats, pts[:, 0], pts[:, 1])).astype(np.float32)


def apply_fx(buf: np.ndarray, events, clip: Clip, tm: TempoMap, na: int, W: int) -> None:
    for ev in sorted(events, key=lambda e: e.beat):
        s0 = tm.beat_to_sample(ev.beat) - na
        s1 = tm.beat_to_sample(min(ev.beat + ev.duration_beats, clip.length_beats)) - na
        if s1 <= 0 or s0 >= W:
            continue
        a, b = max(s0, 0), min(s1, W)
        beat_sec = 60.0 / float(tm.bpm_at(ev.beat))
        wet = wet_envelope(ev, tm, a, b, na)
        for st in ev.chain or [{"fx": ev.fx, "beats": ev.beats, "params": ev.params}]:
            fxm.apply(buf, st["fx"], a, b, beat_sec, st.get("beats", 1.0), wet * st.get("wet", 1.0), st.get("params"))
