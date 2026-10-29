from __future__ import annotations

from ..core.units import SR
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap
from .cache import AudioCache
from .deck import load_audio, render_clip
from .ducking import cross_key, key_clip
from .sampler import Samples


def clip_ducks(model: SetModel, c, tm: TempoMap, cs: int, na: int, nb: int, rows: dict, info, cache: AudioCache, samples: Samples) -> list:
    out = []
    for ev in (e for e in model.ducks if e.target == c.id):
        kid = key_clip(ev)
        if kid is None or kid == c.id:
            out.append((ev, None))
            continue
        kc = model.clip(kid)
        kinfo = info(kc.track)
        ktm = TempoMap(kc, float(kinfo["bpm"]))
        render = lambda w: render_clip(kc, kinfo, load_audio(kc, kinfo, cache), ktm, w, samples=samples)
        out.append((ev, cross_key(ev, c, tm, cs - int(round(rows[kid]["start_sec"] * SR)), ktm.beat_to_sample(kc.length_beats), render, na, nb)))
    return out
