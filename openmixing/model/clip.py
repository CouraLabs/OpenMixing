from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.units import BEAT_DIVISIONS
from .events import FxEvent, LoopEvent, RateEvent, Trigger
from .lane import LANES, LEGACY_LANES, Lane


@dataclass
class Clip:
    id: str
    track: str
    src_start_beat: float = 0.0
    length_beats: float = 64.0
    start: dict[str, Any] = field(default_factory=lambda: {"at_sec": 0.0})
    tempo: Lane = field(default_factory=Lane)
    semitones: float = 0.0
    pitch: Lane = field(default_factory=Lane)
    key_lock: bool = True
    gain_db: float = 0.0
    stems: bool = False
    lanes: dict[str, Lane] = field(default_factory=lambda: {k: Lane(v[0]) for k, v in LANES.items()})
    fx: list[FxEvent] = field(default_factory=list)
    loops: list[LoopEvent] = field(default_factory=list)
    rate: list[RateEvent] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("id", "track", "src_start_beat", "length_beats", "start", "semitones",
                                           "key_lock", "gain_db", "stems", "label")}
        d["tempo"] = self.tempo.points
        d["pitch"] = self.pitch.points
        d["lanes"] = {k: v.points for k, v in self.lanes.items() if v.points}
        d["fx"] = [e.__dict__ for e in self.fx]
        d["loops"] = [e.__dict__ for e in self.loops]
        d["rate"] = [e.__dict__ for e in self.rate]
        d["triggers"] = [e.__dict__ for e in self.triggers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        c = cls(id=d["id"], track=d["track"])
        for k in ("src_start_beat", "length_beats", "start", "semitones", "key_lock", "gain_db", "stems", "label"):
            if k in d:
                setattr(c, k, d[k])
        c.tempo = Lane(0.0, d.get("tempo", []))
        c.pitch = Lane(0.0, d.get("pitch", []))
        for k, pts in d.get("lanes", {}).items():
            for name in LEGACY_LANES.get(k, (k,)):
                c.lanes[name].points = [list(p) for p in pts]
        c.fx = [FxEvent(**e) for e in d.get("fx", [])]
        c.loops = [LoopEvent(**e) for e in d.get("loops", [])]
        c.rate = [RateEvent(**e) for e in d.get("rate", [])]
        c.triggers = [Trigger(**e) for e in d.get("triggers", [])]
        return c

    def validate_loops(self) -> None:
        ls = sorted(self.loops, key=lambda l: l.beat)
        for a, b in zip(ls, ls[1:]):
            if a.beat + a.length_beats > b.beat + 1e-9:
                raise ValueError(f"loops {a.id} and {b.id} overlap")
        for l in ls:
            if not any(abs(l.size_beats - d) < 1e-9 for d in BEAT_DIVISIONS):
                raise ValueError(f"bad loop size {l.size_beats}")

    def validate_rate(self) -> None:
        end = lambda e: e.beat + (e.duration_beats if isinstance(e, RateEvent) else e.length_beats)
        evs = sorted([*self.rate, *self.loops], key=lambda e: e.beat)
        for a, b in zip(evs, evs[1:]):
            if RateEvent in (type(a), type(b)) and end(a) > b.beat + 1e-9:
                raise ValueError(f"{a.id} and {b.id} overlap")
