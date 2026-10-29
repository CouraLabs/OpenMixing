from __future__ import annotations

from typing import Literal

from ...core.units import BAR
from ...model.events import RateEvent, new_id
from ...model.rate import check_curve, preset
from ...plan import RANGES
from ..app import Len, Pos, tool
from ..edit import without
from ..state import LOCK, at_of, clip_of, len_of, save


@tool()
def add_rate(clip: str, at_bar: Pos, bars: Len, kind: Literal["spinback", "reverse", "brake", "custom"] = "spinback",
             curve: list[list[float]] | None = None, slip: bool = False) -> dict:
    """Change the playback rate of a deck for `bars` bars (1/16 .. 32 beats), vinyl style: pitch follows speed. spinback = speed 1 -> 0
    with an ease-in, brake = 1 -> 0 linear, reverse = constant -1 (true backwards playback), custom needs `curve`: [u, speed] points with
    u 0..1 over the event, speed 1 normal, 0 stopped, negative backwards (-4..4); a curve overrides the preset and kind stays a label.
    Without slip the source ends where the curve left it; slip=True lets the track keep running underneath and rejoin in time (the same
    behaviour as a loop roll). Rate events may not overlap each other or a loop."""
    pts = [[float(u), float(v)] for u, v in curve] if curve else preset(kind)
    check_curve(pts, *RANGES["rate_speed"])
    with LOCK:
        c = clip_of(clip)
        at, dur = at_of(c, at_bar), len_of(bars) * BAR
        if not RANGES["rate_beats"][0] <= dur <= RANGES["rate_beats"][1]:
            raise ValueError(f"a rate event lasts between {RANGES['rate_beats'][0]:g} and {RANGES['rate_beats'][1]:g} beats")
        ev = RateEvent(new_id("rt"), at * BAR, dur, pts, slip, kind)
        c.rate.append(ev)
        try:
            c.validate_rate()
        except ValueError:
            c.rate.remove(ev)
            raise
        save()
    return {"clip": clip, "rate_id": ev.id, "ends_at_bar": (ev.beat + dur) / BAR}


@tool()
def remove_rate(clip: str, rate_id: str) -> dict:
    """Remove a rate event (spinback, reverse, brake, custom curve)."""
    with LOCK:
        c = clip_of(clip)
        c.rate = without(c.rate, rate_id, "rate_id")
        save()
    return {"clip": clip, "remaining_rate": [r.id for r in c.rate]}
