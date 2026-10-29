from __future__ import annotations

from dataclasses import dataclass, field, fields

from ..core.errors import OpenMixingError


@dataclass
class Constraints:
    must_include: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    start_with: str | None = None
    end_with: str | None = None
    fixed_positions: dict[int, str] = field(default_factory=dict)
    must_not_follow: list[list[str]] = field(default_factory=list)
    max_bpm_jump_percent: float | None = None
    max_key_distance: int | None = None
    min_gap_between_artist_repeats: int = 0
    avoid_played_within_days: float | None = None
    max_tracks: int | None = None
    only_with_stems: bool = False
    crate: str | None = None
    duration_max: float | None = None
    duration_tolerance_percent: float = 5.0
    min_play_bars: int = 32


def parse(raw: dict | None, resolve) -> Constraints:
    raw = dict(raw or {})
    bad = [k for k in raw if k not in {f.name for f in fields(Constraints)}]
    if bad:
        raise OpenMixingError("invalid_value", f"unknown constraints {bad}", "constraints", f"use {[f.name for f in fields(Constraints)]}")
    for k in ("must_include", "must_exclude"):
        raw[k] = [resolve(r) for r in raw.get(k, [])]
    for k in ("start_with", "end_with"):
        if raw.get(k):
            raw[k] = resolve(raw[k])
    raw["fixed_positions"] = {int(i): resolve(r) for i, r in raw.get("fixed_positions", {}).items()}
    raw["must_not_follow"] = [[resolve(a), resolve(b)] for a, b in raw.get("must_not_follow", [])]
    return Constraints(**raw)


def pinned(c: Constraints) -> set[str]:
    return {*c.must_include, *c.fixed_positions.values(), *(x for x in (c.start_with, c.end_with) if x)}
