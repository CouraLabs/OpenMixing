from __future__ import annotations

import numpy as np

from ..core.decode import decode
from ..dsp import sample
from ..model.sampler import SampleSlot
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap

Samples = dict[str, tuple[SampleSlot, np.ndarray]]


def load_samples(model: SetModel) -> Samples:
    used = {t.slot for c in model.clips for t in c.triggers if t.kind == "sample"}
    return {k: (s, decode(s.path)) for k, s in model.sampler.items() if k in used}


def choke_cuts(triggers: list, samples: Samples, tm: TempoMap) -> dict[str, int]:
    group = lambda t: samples[t.slot][0].choke_group if t.slot in samples else None
    hits = [t for t in sorted(triggers, key=lambda t: t.start) if t.kind == "sample" and group(t)]
    out: dict[str, int] = {}
    for i, a in enumerate(hits):
        for b in hits[i + 1 :]:
            if b.params.get("choke") and group(b) == group(a) and b.start < a.start + a.duration_beats:
                out[a.id] = tm.beat_to_sample(b.start)
                break
    return out


def sample_voice(t, tm: TempoMap, samples: Samples, n: int, cut_at: int | None, s0: int) -> np.ndarray:
    if t.slot not in samples:
        raise ValueError(f"trigger {t.id} plays sampler slot {t.slot!r} which is not loaded: call load_sample")
    slot, audio = samples[t.slot]
    p = t.params
    y = sample.voice(audio, slot.kind, slot.root_bpm, float(tm.bpm_at(t.start)), n, slot.pitch_semitones + p.get("pitch_semitones", 0.0),
                     int(p.get("fit", 0)))
    if cut_at is not None:
        sample.cut(y, cut_at - s0)
    return y * np.float32(10 ** ((slot.gain_db + p.get("velocity", 0.0)) / 20))
