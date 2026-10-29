from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clip import Clip
from .events import DuckEvent
from .sampler import SampleSlot


@dataclass
class SetModel:
    name: str
    clips: list[Clip] = field(default_factory=list)
    notes: str = ""
    version: int = 0
    intent: dict[str, Any] = field(default_factory=dict)
    ducks: list[DuckEvent] = field(default_factory=list)
    sampler: dict[str, SampleSlot] = field(default_factory=dict)

    def clip(self, cid: str) -> Clip:
        for c in self.clips:
            if c.id == cid:
                return c
        raise KeyError(f"no clip {cid!r} in set {self.name!r}; clips: {[c.id for c in self.clips]}")

    def to_dict(self) -> dict:
        d = {"name": self.name, "notes": self.notes, "version": self.version, "clips": [c.to_dict() for c in self.clips]}
        if self.intent:
            d["intent"] = self.intent
        if self.ducks:
            d["ducks"] = [e.__dict__ for e in self.ducks]
        if self.sampler:
            d["sampler"] = {k: v.to_dict() for k, v in self.sampler.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SetModel":
        return cls(d["name"], [Clip.from_dict(c) for c in d.get("clips", [])], d.get("notes", ""), d.get("version", 0), d.get("intent", {}),
                   [DuckEvent(**e) for e in d.get("ducks", [])], {k: SampleSlot(**v) for k, v in d.get("sampler", {}).items()})

    def save(self, path: Path) -> None:
        self.version += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SetModel":
        return cls.from_dict(json.loads(path.read_text()))
