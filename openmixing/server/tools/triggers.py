from __future__ import annotations

from typing import Literal

from ...core.errors import OpenMixingError
from ...core.units import BAR, parse_beats
from ...dsp import sample, synth
from ...model.events import Trigger, new_id
from ...model.roll import expand_roll
from ...plan import RANGES
from ..app import Len, Pos, tool
from ..edit import without
from ..state import LOCK, at_of, clip_of, len_of, save

Kind = Literal[*synth.KINDS]


def checked_params(kind: str, params: dict | None) -> dict:
    spec = sample.param_spec(kind)
    for k, v in (params or {}).items():
        if k not in spec:
            raise OpenMixingError("unknown_param", f"{kind} has no parameter {k!r}", f"params.{k}", f"allowed: {sorted(spec)}")
        if not spec[k][1] <= v <= spec[k][2]:
            raise OpenMixingError("out_of_range", f"{k} {v} is outside [{spec[k][1]}, {spec[k][2]}]", f"params.{k}",
                                  f"use a value between {spec[k][1]} and {spec[k][2]}")
    return dict(params or {})


def checked_gain(gain_db: float) -> float:
    lo, hi = RANGES["gain_db"]
    if not lo <= gain_db <= hi:
        raise OpenMixingError("out_of_range", f"gain_db {gain_db} is outside [{lo}, {hi}]", "gain_db", f"use a value between {lo} and {hi}")
    return float(gain_db)


@tool()
def list_generators() -> dict:
    """Synthesized transition material for add_trigger / add_roll: kinds, their default alignment and their numeric params (default, min, max).
    noise `color` is 0 white, 1 pink, 2 brown. Levels are relative to the deck: gain_db 0 puts the generator's peak at the deck's typical
    peak (its RMS over the preceding 4 beats + 9 dB; -18 dBFS RMS if the deck is silent). Output is seeded from the trigger id."""
    return {"kinds": synth.describe(), "noise_colors": ["white", "pink", "brown"], "roll_curves": ["log", "linear"]}


@tool()
def add_trigger(clip: str, kind: Kind, at_bar: Pos, bars: Len = 1.0, align: Literal["start", "end"] | None = None, gain_db: float = -6.0,
                params: dict | None = None) -> dict:
    """Add a synthesized riser / downlifter / impact / crash / snare / noise bed to a deck, post-fader and before the deck fx (so an echo can
    throw it). align="end" makes it finish exactly at `at_bar` (a riser peaking on the drop; default for riser), "start" makes it begin
    there (impacts, crashes; default for everything else). It lasts `bars`. gain_db is relative to the deck level (see list_generators);
    `params` are numbers from list_generators. Re-rendering gives identical audio."""
    p, g = checked_params(kind, params), checked_gain(gain_db)
    with LOCK:
        c = clip_of(clip)
        align = align or synth.KINDS[kind]["align"]
        t = Trigger(new_id("tg"), kind, at_of(c, at_bar) * BAR, len_of(bars) * BAR, align, g, p)
        if t.start < -1e-9 or t.start > c.length_beats:
            raise OpenMixingError("out_of_range", f"the {kind} would start at beat {t.start:g}, outside the clip", "at_bar",
                                  "move it so the whole generator starts inside the clip")
        c.triggers.append(t)
        save()
    return {"clip": clip, "trigger_id": t.id, "starts_at_bar": t.start / BAR, "ends_at_bar": (t.start + t.duration_beats) / BAR}


@tool()
def add_roll(clip: str, at_bar: Pos, bars: Len, kind: Kind = "snare", start_div: str | float = "1/2", end_div: str | float = "1/16",
             curve: Literal["log", "linear"] = "log", gain_db: float = -12.0, gain_end_db: float = 0.0, params: dict | None = None) -> dict:
    """Percussion roll: a run of one-shot triggers (default snare) from `at_bar` for `bars`, accelerating from one hit every start_div beats
    to one every end_div beats (tempo-sync values 1/16 .. 32; curve log = constant ratio, linear = constant change) with the level
    moving from gain_db to gain_end_db. Expands to ordinary triggers you can remove one by one with remove_trigger."""
    sd, ed = parse_beats(start_div), parse_beats(end_div)
    p, g0, g1 = checked_params(kind, params), checked_gain(gain_db), checked_gain(gain_end_db)
    with LOCK:
        c = clip_of(clip)
        at, length = at_of(c, at_bar) * BAR, len_of(bars) * BAR
        if at + length > c.length_beats + 1e-9:
            raise OpenMixingError("out_of_range", "the roll runs past the end of the clip", "bars", "shorten it")
        ts = expand_roll(at, length, sd, ed, curve, g0, g1, kind, p)
        c.triggers.extend(ts)
        save()
    return {"clip": clip, "trigger_ids": [t.id for t in ts], "hits": len(ts)}


@tool()
def remove_trigger(clip: str, trigger_id: str) -> dict:
    """Remove a trigger (riser, impact, one hit of a roll ...)."""
    with LOCK:
        c = clip_of(clip)
        c.triggers = without(c.triggers, trigger_id, "trigger_id")
        save()
    return {"clip": clip, "remaining_triggers": len(c.triggers)}
