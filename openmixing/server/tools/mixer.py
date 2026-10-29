from __future__ import annotations

from typing import Literal

import numpy as np

from ... import mixing
from ...core.errors import OpenMixingError
from ...core.units import BAR
from ...model.lane import LANES
from ..app import Len, Pos, tool
from ..edit import automate
from ..state import LOCK, at_of, clip_of, conv_beat, cur, info, len_of, save
from ..views import describe_events, jsonable, lane_name


@tool()
def set_eq(clip: str, band: Literal["low", "mid", "high", "all"], at_bar: Pos, gain_db: float | Literal["kill"] = 0.0,
           ramp_bars: Len = 0.0) -> dict:
    """Move a 3-band EQ knob on a deck (low < 200 Hz, mid 200-2500 Hz, high > 2.5 kHz). gain_db from -80 ("kill") to +6; 0 = flat.
    ramp_bars = 0 snaps (de-clicked over ~10 ms). Works on the playing deck or the incoming one."""
    g = -80.0 if gain_db == "kill" else float(gain_db)
    if not -80.0 <= g <= 6.0:
        raise ValueError("gain_db must be in [-80 (kill), +6]")
    bands = ["low", "mid", "high"] if band == "all" else [band]
    return automate(clip, at_bar, ramp_bars, {lane_name(b): g for b in bands}, "eq_")


@tool()
def set_filter(clip: str, at_bar: Pos, value: float, ramp_bars: Len = 0.0) -> dict:
    """DJ filter knob: -1 = low-pass fully closed (muffled), 0 = off, +1 = high-pass fully closed (thin). Sweep with ramp_bars."""
    return automate(clip, at_bar, ramp_bars, {"filter": float(np.clip(value, -1, 1))}, "filter")


@tool()
def set_volume(clip: str, at_bar: Pos, level: float, ramp_bars: Len = 0.0) -> dict:
    """Channel fader (0-1.5, 1 = unity). It sits before the fx, so echo / reverb tails keep ringing when it closes."""
    return automate(clip, at_bar, ramp_bars, {"volume": float(np.clip(level, 0, 1.5))}, "volume")


@tool()
def crossfade(out_clip: str, in_clip: str, at_bar: Pos, bars: Len, curve: Literal["equal_power", "linear", "slow_in", "cut"] = "equal_power") -> dict:
    """Move the faders between two decks. at_bar is a bar of out_clip; the incoming clip must overlap it in time (see apply_transition
    for a full blend with EQ). Does not touch EQ, so combine with set_eq to avoid bass clashes."""
    with LOCK:
        o, i = clip_of(out_clip), clip_of(in_clip)
        at_bar, bars = at_of(o, at_bar), len_of(bars)
        fn_in, fn_out = {
            "equal_power": (lambda p: np.sin(p * np.pi / 2), lambda p: np.cos(p * np.pi / 2)),
            "linear": (lambda p: p, lambda p: 1 - p),
            "slow_in": (lambda p: p ** 2, lambda p: np.cos(p * np.pi / 2)),
            "cut": (lambda p: float(p >= 0.5), lambda p: float(p < 0.5)),
        }[curve]
        b0 = at_bar * BAR
        ib0 = conv_beat(o, b0, i)
        ib1 = conv_beat(o, b0 + bars * BAR, i)
        if not i.lanes["volume"].points:
            i.lanes["volume"].hold(0.0)
        o.lanes["volume"].curve(b0, bars * BAR, fn_out, 4)
        i.lanes["volume"].curve(ib0, ib1 - ib0, fn_in, 4)
        save()
    return {"out": out_clip, "in": in_clip, "incoming_beat_range": [round(ib0, 2), round(ib1, 2)]}


@tool()
def clear_automation(clip: str, lane: Literal["eq_low", "eq_mid", "eq_high", "filter", "volume", "stem_vocals", "stem_drums", "stem_bass",
                                              "stem_other", "tempo", "pitch", "all"] = "all",
                     from_bar: Pos = 0.0, to_bar: Pos | None = None) -> dict:
    """Reset a lane (or all lanes, fx, loops and rate events with lane="all") to neutral between two bars (default: the whole clip)."""
    with LOCK:
        c = clip_of(clip)
        from_bar = at_of(c, from_bar, "from_bar")
        to_bar = None if to_bar is None else at_of(c, to_bar, "to_bar")
        a, b = from_bar * BAR, (c.length_beats if to_bar is None else to_bar * BAR)
        if a > b:
            raise OpenMixingError("invalid_range", f"from_bar {from_bar:g} is after to_bar", "from_bar", "use from_bar <= to_bar")
        whole = from_bar == 0 and to_bar is None
        names = list(LANES) if lane == "all" else [lane]
        for n in names:
            if n in ("tempo", "pitch"):
                lane = getattr(c, n)
                lane.points = [] if whole else [p for p in lane.points if not a <= p[0] <= b]
            elif whole:
                c.lanes[n].points = []
            else:
                c.lanes[n].clear(a, b)
        if lane == "all":
            c.fx = [e for e in c.fx if not (a <= e.beat <= b)]
            c.loops = [l for l in c.loops if not (a <= l.beat <= b)]
            c.rate = [r for r in c.rate if not (a <= r.beat <= b)]
            if whole:
                c.tempo.points, c.pitch.points = [], []
        save()
    return {"clip": clip, "events": describe_events(c)}


@tool()
def apply_transition(out_clip: str, in_clip: str, style: Literal["bass_swap", "long_blend", "filter_fade", "echo_out", "drop_swap", "loop_roll_out",
                                                                 "stem_blend"] = "bass_swap",
                     bars: Len = 16.0, at_bar: Pos | None = None, tempo_after: Literal["keep", "native"] = "keep",
                     trim_out: bool = True, gap_beats: float = 0.0) -> dict:
    """Mix in_clip over out_clip like a DJ: places and tempo-syncs the incoming deck on a bar of the outgoing one and writes the EQ /
    fader / filter / fx / loop moves. Styles: bass_swap (classic), long_blend (32+ bars, harmonic), filter_fade, echo_out, drop_swap,
    loop_roll_out, stem_blend (both tracks need separate_stems: vocals out, harmonics cross, drums+bass swap, vocals in). at_bar = bar of
    out_clip where in_clip's bar 0 lands (default: `bars` before out_clip ends, so the blend runs to the
    end). Overwrites the transition's automation; trim_out ends the outgoing clip when the blend finishes. gap_beats (echo_out, drop_swap):
    the outgoing deck cuts that many beats before the incoming hit, a breath of silence (the echo tail still rings). Read `audit_set` afterwards."""
    with LOCK:
        o, i = clip_of(out_clip), clip_of(in_clip)
        at_bar = None if at_bar is None else at_of(o, at_bar)
        r = mixing.apply_template(cur(), info, o, i, style, at_bar, len_of(bars), tempo_after, trim_out, gap_beats)
        save()
    r["templates"] = mixing.TEMPLATES[style]
    return jsonable(r)
