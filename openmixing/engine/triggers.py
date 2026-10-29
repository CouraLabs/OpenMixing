from __future__ import annotations

import numpy as np

from ..dsp import synth
from ..model.clip import Clip
from ..model.tempo import TempoMap
from .sampler import Samples, choke_cuts, sample_voice

CREST_DB = 9.0
SILENT_RMS_DB = -18.0
SILENCE = 1e-4
LOOKBACK_BEATS = 4.0


def deck_rms_db(dry: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(lo, 0), min(hi, dry.shape[1])
    rms = float(np.sqrt((dry[:, lo:hi].astype(np.float64) ** 2).mean())) if hi > lo else 0.0
    return 20 * np.log10(rms) if rms > SILENCE else SILENT_RMS_DB


def add_triggers(buf: np.ndarray, dry: np.ndarray, clip: Clip, tm: TempoMap, na: int, W: int, samples: Samples) -> None:
    cuts = choke_cuts(clip.triggers, samples, tm)
    for t in sorted(clip.triggers, key=lambda t: t.start):
        s0, s1 = tm.beat_to_sample(t.start), tm.beat_to_sample(t.start + t.duration_beats)
        r0, r1 = s0 - na, s1 - na
        if s1 - s0 < 2 or r1 <= 0 or r0 >= W:
            continue
        ref = deck_rms_db(dry, tm.beat_to_sample(max(t.start - LOOKBACK_BEATS, 0.0)) - na, r0)
        raw = sample_voice(t, tm, samples, s1 - s0, cuts.get(t.id), s0) if t.kind == "sample" else synth.generate(t.kind, t.id, s1 - s0, t.params)
        y = raw * np.float32(10 ** ((ref + CREST_DB + t.gain_db) / 20))
        lo, hi = max(r0, 0), min(r1, buf.shape[1])
        buf[:, lo:hi] += y[:, lo - r0 : hi - r0]
