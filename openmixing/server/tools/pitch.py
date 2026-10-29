from __future__ import annotations

from typing import Literal

from ...core.units import BAR
from ...mixing import plan_key_match
from ...model.curves import write
from ..app import Curve, Len, Pos, tool
from ..state import LOCK, at_of, clip_of, info, len_of, save
from ..views import describe_events


@tool()
def set_pitch(clip: str, at_bar: Pos, semitones: float, ramp_bars: Len = 0.0, curve: Curve = "linear", exponent: float = 2.0) -> dict:
    """Pitch lane of a deck: shift the pitch by `semitones` (-12..12, fractions allowed) at a bar, optionally gliding along a curve. It adds
    to the clip's constant `semitones`; the tempo is untouched. Glides are rendered in steps of at most 0.25 semitone."""
    if not -12.0 <= semitones <= 12.0:
        raise ValueError("semitones must be in [-12, 12]")
    with LOCK:
        c = clip_of(clip)
        write(c.pitch, at_of(c, at_bar) * BAR, float(semitones), len_of(ramp_bars, "ramp_bars") * BAR, curve, exponent)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "pitch" in e]}


@tool()
def match_key(clip: str, to_clip: str, mode: Literal["nearest_compatible", "exact"] = "nearest_compatible", max_shift: int = 3,
              prefer: Literal["incoming", "outgoing", "split"] = "incoming", apply: bool = True, over_bars: Len | None = None) -> dict:
    """Pitch-shift `clip` (incoming) so its key fits `to_clip` (outgoing reference), using the Camelot wheel (+1 semitone = +7 steps) and the
    tuning offset of both tracks. nearest_compatible stops at the smallest shift that is same / adjacent / relative, exact needs the same key.
    prefer picks which deck takes the shift (split halves it on both). Proposals with key confidence < 0.4 are returned flagged low_confidence
    and not applied. over_bars glides the pitch lane from 0 over that many bars from the clip start instead of a hard transpose."""
    if not 0 <= max_shift <= 12:
        raise ValueError("max_shift must be in [0, 12]")
    with LOCK:
        c, o = clip_of(clip), clip_of(to_clip)
        r = plan_key_match(info(o.track), info(c.track), max_shift, mode, prefer)
        r["applied"] = apply and not r["low_confidence"]
        if r["applied"]:
            if prefer != "incoming":
                o.semitones = r["semitones_out"]
            if over_bars is None:
                c.semitones = r["semitones_in"]
            else:
                c.semitones = 0.0
                write(c.pitch, 0.0, r["semitones_in"], len_of(over_bars, "over_bars") * BAR)
            c.key_lock = True
            save()
    return r
