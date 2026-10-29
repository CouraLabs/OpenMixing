from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


@dataclass
class FxEvent:
    id: str
    fx: str
    beat: float
    duration_beats: float
    beats: float = 1.0
    wet: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    stem: str | None = None
    chain: list[dict[str, Any]] | None = None
    wet_curve: list[list[float]] | None = None
    group: str | None = None


@dataclass
class DuckEvent:
    id: str
    target: str
    key: str
    beat: float
    duration_beats: float
    depth_db: float = 6.0
    threshold_db: float = -30.0
    attack_ms: float = 5.0
    release_ms: float = 120.0
    stem: str | None = None


@dataclass
class LoopEvent:
    id: str
    beat: float
    size_beats: float
    length_beats: float
    roll: bool = False
    group: str | None = None


@dataclass
class RateEvent:
    id: str
    beat: float
    duration_beats: float
    curve: list[list[float]]
    slip: bool = False
    kind: str = "custom"


@dataclass
class Trigger:
    id: str
    kind: str
    beat: float
    duration_beats: float = 4.0
    align: str = "start"
    gain_db: float = -6.0
    params: dict[str, Any] = field(default_factory=dict)
    slot: str | None = None
    group: str | None = None

    @property
    def start(self) -> float:
        return self.beat - self.duration_beats if self.align == "end" else self.beat
