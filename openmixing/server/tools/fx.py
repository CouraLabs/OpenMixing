from __future__ import annotations

from typing import Literal

import numpy as np

from ...core.errors import OpenMixingError
from ...core.units import BAR, BEAT_DIVISIONS, BEAT_LABELS, parse_beats
from ...dsp import fx as fxm
from ...model.events import FxEvent, LoopEvent, new_id
from ...plan.ranges import MAX_CHAIN
from ..app import Beats, Len, Pos, tool
from ..edit import without
from ..state import LOCK, at_of, clip_of, len_of, save


@tool()
def list_fx() -> dict:
    """Available effects, what the beats multiplier controls for each, and their extra params with defaults."""
    return {"beats_values": [BEAT_LABELS[b] for b in BEAT_DIVISIONS],
            "fx": {n: {"mode": s.mode, "description": s.description, "beats_controls": s.uses_beats or "(not tempo-synced)", "params": s.params}
                   for n, s in fxm.CATALOG.items()}}


@tool()
def add_fx(clip: str, fx: str, at_bar: Pos, bars: Len, beats: Beats = 1.0, wet: float = 1.0, params: dict | None = None,
           stem: Literal["vocals", "drums", "bass", "other"] | None = None, chain: list[dict] | None = None,
           wet_points: list[list[float]] | None = None) -> dict:
    """Switch a tempo-synced effect on for `bars` bars from `at_bar` (see list_fx). beats in {1/16 ... 32} sets the sync division;
    wet 0-1 is the FX dry/wet. Tails of echo / reverb ring out after the effect ends. stem: apply the effect to one stem only
    (e.g. echo the vocals, keep drums dry); the clip must play its stems (stems=True after separate_stems).
    chain: list of up to 6 steps {fx, beats, wet, params} run in series over the same window (fx is then ignored; wet scales all steps).
    wet_points: [[bar, wet], ...] from the start of the effect, linear between, multiplies wet (e.g. [[0, 0], [4, 1]] fades it in)."""
    steps = [{"fx": s["fx"], "beats": parse_beats(s.get("beats", 1.0)), "wet": float(np.clip(s.get("wet", 1.0), 0, 1)),
              "params": s.get("params") or {}} for s in chain or []]
    if len(steps) > MAX_CHAIN:
        raise ValueError(f"a chain has at most {MAX_CHAIN} steps")
    bad = [n for n in [s["fx"] for s in steps] or [fx] if n not in fxm.CATALOG]
    if bad:
        raise ValueError(f"unknown fx {bad[0]!r}; one of {list(fxm.CATALOG)}")
    curve = [[len_of(b) * BAR, float(np.clip(w, 0, 1))] for b, w in wet_points or []]
    if any(q[0] >= p[0] for p, q in zip(curve[1:], curve)):
        raise ValueError("wet_points bars must increase")
    with LOCK:
        c = clip_of(clip)
        at_bar, bars = at_of(c, at_bar), len_of(bars)
        if stem and not c.stems:
            raise OpenMixingError("stems_off", f"clip {clip} plays the full mix, so a {stem}-only effect has nothing to act on", "stem",
                                "call separate_stems, then edit_clip(stems=True)")
        e = FxEvent(new_id("fx"), "chain" if steps else fx, at_bar * BAR, bars * BAR, parse_beats(beats), float(np.clip(wet, 0, 1)), params or {},
                    stem, steps or None, curve or None)
        c.fx.append(e)
        save()
    return {"clip": clip, "fx_id": e.id}


@tool()
def remove_fx(clip: str, fx_id: str) -> dict:
    """Remove an fx event."""
    with LOCK:
        c = clip_of(clip)
        c.fx = without(c.fx, fx_id, "fx_id")
        save()
    return {"clip": clip, "remaining_fx": [e.id for e in c.fx]}


@tool()
def add_loop(clip: str, at_bar: Pos, size_beats: Beats, bars: Len | None = None, repeats: int | None = None, roll: bool = False) -> dict:
    """Engage a loop on the deck at `at_bar` (output bar; may be fractional, e.g. 31.75 for the last beat of bar 31).

    size_beats in {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32}: the loop repeats the next size_beats of the track. Stay in the loop for
    `bars` bars or `repeats` repetitions. roll=True is a loop roll: the track keeps running underneath and playback resumes where
    it would have been (a stutter); otherwise playback continues from where the loop ended. Loops may not overlap."""
    size = parse_beats(size_beats)
    if (bars is None) == (repeats is None):
        raise ValueError("give exactly one of bars / repeats")
    with LOCK:
        c = clip_of(clip)
        at_bar = at_of(c, at_bar)
        length = len_of(bars) * BAR if bars is not None else size * repeats
        lp = LoopEvent(new_id("lp"), at_bar * BAR, size, length, roll)
        c.loops.append(lp)
        try:
            c.validate_loops()
        except ValueError:
            c.loops.remove(lp)
            raise
        save()
    return {"clip": clip, "loop_id": lp.id, "ends_at_bar": (lp.beat + length) / BAR}


@tool()
def remove_loop(clip: str, loop_id: str) -> dict:
    """Release/remove a loop."""
    with LOCK:
        c = clip_of(clip)
        c.loops = without(c.loops, loop_id, "loop_id")
        save()
    return {"clip": clip, "remaining_loops": [l.id for l in c.loops]}
