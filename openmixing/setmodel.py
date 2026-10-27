"""The set: an ordered timeline of deck 'clips', each with DJ-style automation.

Everything a DJ does live is an event on a clip, positioned in *output beats since the clip started*
(what the DJ hears on that deck): tempo moves, 3-band EQ, the filter knob, the channel fader, tempo-synced
fx and loops. Clips are placed on the set timeline either at an absolute time or anchored to a beat of
another clip, so moving or re-tempoing one deck keeps the mix phase-locked.
"""
from __future__ import annotations

import bisect
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import BEAT_DIVISIONS, SR, parse_beats

LANES = {
    # name: (neutral value, min, max, unit)
    "eq_low": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "eq_mid": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "eq_high": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "filter": (0.0, -1.0, 1.0, "-1 = full low-pass, 0 = off, +1 = full high-pass"),
    "volume": (1.0, 0.0, 1.5, "channel fader (linear)"),
    # per-stem faders, used when the clip plays its stems (Clip.stems); they feed the 3-band EQ / filter / fader above
    "stem_vocals": (1.0, 0.0, 1.5, "vocals stem level (linear)"),
    "stem_drums": (1.0, 0.0, 1.5, "drums stem level (linear)"),
    "stem_bass": (1.0, 0.0, 1.5, "bass stem level (linear)"),
    "stem_other": (1.0, 0.0, 1.5, "other (synths, pads, keys, guitars) stem level (linear)"),
}
STEM_LANES = {"vocals": "stem_vocals", "drums": "stem_drums", "bass": "stem_bass", "other": "stem_other"}
# lanes / stems of sets saved with the old 3-stem layout: drums_bass drove drums and bass together, harmonic became other
LEGACY_LANES = {"stem_drums_bass": ("stem_drums", "stem_bass"), "stem_harmonic": ("stem_other",)}
LEGACY_STEM_NAMES = {"drums_bass": ("drums", "bass"), "harmonic": ("other",)}
MIN_RAMP = 0.02  # beats; instant changes get this ramp so they never click


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


@dataclass
class Lane:
    """Piecewise-linear automation: sorted [beat, value] points; before the first point the first value holds."""

    neutral: float = 0.0
    points: list[list[float]] = field(default_factory=list)

    def value_at(self, beat: float) -> float:
        if not self.points:
            return self.neutral
        xs = [p[0] for p in self.points]
        i = bisect.bisect_right(xs, beat)
        if i == 0:
            return self.points[0][1]
        if i == len(xs):
            return self.points[-1][1]
        (x0, y0), (x1, y1) = self.points[i - 1], self.points[i]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (beat - x0) / (x1 - x0)

    def set(self, beat: float, value: float, ramp: float = 0.0) -> None:
        """Hold the current value until `beat`, then move to `value` over `ramp` beats."""
        ramp = max(ramp, MIN_RAMP)
        before = self.value_at(beat)
        self.points = [p for p in self.points if p[0] < beat or p[0] > beat + ramp]
        self.points += [[beat, before], [beat + ramp, value]]
        self.points.sort(key=lambda p: p[0])

    def hold(self, value: float) -> None:
        """Constant `value` from the start (replaces all points)."""
        self.points = [[0.0, value]]

    def curve(self, beat: float, length: float, fn, steps_per_beat: int = 2) -> None:
        """Write fn(p) for p in 0..1 over [beat, beat+length] as dense points (replaces points in that range)."""
        n = max(int(round(length * steps_per_beat)), 1)
        self.points = [p for p in self.points if p[0] < beat or p[0] > beat + length]
        self.points += [[beat + length * i / n, float(fn(i / n))] for i in range(n + 1)]
        self.points.sort(key=lambda p: p[0])

    def clear(self, a: float, b: float) -> None:
        """Back to neutral over [a, b], dropping every point inside."""
        self.points = [p for p in self.points if p[0] < a or p[0] > b]
        self.points += [[a, self.neutral], [b, self.neutral]]
        self.points.sort(key=lambda p: p[0])

    def is_neutral(self) -> bool:
        return all(abs(p[1] - self.neutral) < 1e-9 for p in self.points)

    def sample(self, beats: np.ndarray) -> np.ndarray:
        if not self.points:
            return np.full(beats.shape, self.neutral, np.float32)
        xs, ys = zip(*self.points)
        return np.interp(beats, xs, ys).astype(np.float32)


@dataclass
class FxEvent:
    id: str
    fx: str
    beat: float
    duration_beats: float
    beats: float = 1.0           # tempo-sync multiplier
    wet: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    stem: str | None = None      # apply to this stem only (vocals / drums / bass / other); needs a clip with stems=True


@dataclass
class LoopEvent:
    id: str
    beat: float                  # output beat at which the loop is engaged
    size_beats: float            # loop size, one of BEAT_DIVISIONS
    length_beats: float          # how long the loop stays engaged (output beats)
    roll: bool = False           # loop roll: the track keeps running underneath and resumes in time on exit


@dataclass
class Clip:
    id: str
    track: str
    src_start_beat: float = 0.0  # track beat (0 = first downbeat) where this deck starts playing
    length_beats: float = 64.0   # output beats played (loops included)
    start: dict[str, Any] = field(default_factory=lambda: {"at_sec": 0.0})
    tempo: Lane = field(default_factory=Lane)      # bpm automation; empty = track's native bpm
    semitones: float = 0.0
    key_lock: bool = True
    gain_db: float = 0.0
    stems: bool = False          # play the separated stems (summed through their stem_* faders) instead of the full mix
    lanes: dict[str, Lane] = field(default_factory=lambda: {k: Lane(v[0]) for k, v in LANES.items()})
    fx: list[FxEvent] = field(default_factory=list)
    loops: list[LoopEvent] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("id", "track", "src_start_beat", "length_beats", "start", "semitones",
                                           "key_lock", "gain_db", "stems", "label")}
        d["tempo"] = self.tempo.points
        d["lanes"] = {k: v.points for k, v in self.lanes.items() if v.points}
        d["fx"] = [e.__dict__ for e in self.fx]
        d["loops"] = [e.__dict__ for e in self.loops]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        c = cls(id=d["id"], track=d["track"])
        for k in ("src_start_beat", "length_beats", "start", "semitones", "key_lock", "gain_db", "stems", "label"):
            if k in d:
                setattr(c, k, d[k])
        c.tempo = Lane(0.0, d.get("tempo", []))
        for k, pts in d.get("lanes", {}).items():
            for name in LEGACY_LANES.get(k, (k,)):
                c.lanes[name].points = [list(p) for p in pts]
        c.fx = [FxEvent(**e) for e in d.get("fx", [])]
        c.loops = [LoopEvent(**e) for e in d.get("loops", [])]
        return c

    def validate_loops(self) -> None:
        ls = sorted(self.loops, key=lambda l: l.beat)
        for a, b in zip(ls, ls[1:]):
            if a.beat + a.length_beats > b.beat + 1e-9:
                raise ValueError(f"loops {a.id} and {b.id} overlap")
        for l in ls:
            if not any(abs(l.size_beats - d) < 1e-9 for d in BEAT_DIVISIONS):
                raise ValueError(f"bad loop size {l.size_beats}")


@dataclass
class SetModel:
    name: str
    clips: list[Clip] = field(default_factory=list)
    notes: str = ""
    version: int = 0
    intent: dict[str, Any] = field(default_factory=dict)   # what the plan expects the render to show, see validate.INTENT_KEYS

    def clip(self, cid: str) -> Clip:
        for c in self.clips:
            if c.id == cid:
                return c
        raise KeyError(f"no clip {cid!r} in set {self.name!r}; clips: {[c.id for c in self.clips]}")

    def to_dict(self) -> dict:
        d = {"name": self.name, "notes": self.notes, "version": self.version, "clips": [c.to_dict() for c in self.clips]}
        if self.intent:
            d["intent"] = self.intent
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SetModel":
        return cls(d["name"], [Clip.from_dict(c) for c in d.get("clips", [])], d.get("notes", ""), d.get("version", 0), d.get("intent", {}))

    def save(self, path: Path) -> None:
        self.version += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SetModel":
        return cls.from_dict(json.loads(path.read_text()))


class TempoMap:
    """Output beat <-> clip-local seconds, for a clip whose bpm is piecewise-linear in output beats."""

    RES = 32  # grid points per beat

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
    """Absolute start (seconds on the set timeline) of every clip. native_bpm(track_id) -> bpm."""
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
