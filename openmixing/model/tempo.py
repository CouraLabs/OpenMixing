from __future__ import annotations

import numpy as np

from ..core.units import SR
from .clip import Clip
from .setmodel import SetModel


class TempoMap:

    RES = 32

    def __init__(self, clip: Clip, native_bpm: float):
        n = int(np.ceil(clip.length_beats * self.RES)) + 2
        self.beats = np.arange(n) / self.RES
        if clip.tempo.points:
            xs, ys = zip(*clip.tempo.points)
            self.bpm = np.interp(self.beats, xs, ys)
        else:
            self.bpm = np.full(n, native_bpm)
        sec_per_beat = 60.0 / self.bpm
        mid = (sec_per_beat[1:] + sec_per_beat[:-1]) / 2
        self.secs = np.concatenate([[0.0], np.cumsum(mid / self.RES)])
        self.native = native_bpm

    def beat_to_sec(self, beat):
        return np.interp(beat, self.beats, self.secs)

    def sec_to_beat(self, sec):
        return np.interp(sec, self.secs, self.beats)

    def beat_to_sample(self, beat):
        sec = self.beat_to_sec(beat) * SR
        return np.round(sec).astype(np.int64) if np.ndim(beat) else int(round(float(sec)))

    def bpm_at(self, beat):
        return np.interp(beat, self.beats, self.bpm)


def resolve_starts(model: SetModel, native_bpm) -> dict[str, float]:
    out: dict[str, float] = {}
    pending = {c.id: c for c in model.clips}
    while pending:
        progressed = False
        for cid, c in list(pending.items()):
            s = c.start
            off = s.get("offset_ms", 0.0) / 1000.0
            if "at_sec" in s:
                out[cid] = float(s["at_sec"]) + off
            elif s.get("anchor") in out:
                a = model.clip(s["anchor"])
                tm = TempoMap(a, native_bpm(a.track))
                out[cid] = out[a.id] + float(tm.beat_to_sec(s["anchor_beat"])) + off
            elif s.get("anchor") not in pending:
                raise ValueError(f"clip {cid} is anchored to unknown clip {s.get('anchor')!r}")
            else:
                continue
            del pending[cid]
            progressed = True
        if not progressed:
            raise ValueError(f"anchor cycle among clips {list(pending)}")
    return out
