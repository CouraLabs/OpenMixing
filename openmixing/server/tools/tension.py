from __future__ import annotations

from ...core.units import BAR
from ...mixing import ELEMENTS, TECHNIQUES, build_and_release, existing_gap, remove_group
from ...mixing.techniques import TENSION_MAP
from ...model.events import new_id
from ..app import Len, Pos, tool
from ..state import LOCK, at_of, clip_of, len_of, lib, save


@tool()
def list_techniques() -> dict:
    """Named techniques that expand into ordinary lanes, fx, loops and triggers (tagged group=tension:<id>), their build elements and the
    fixed mapping from a tension level 0..1 to each element (TENSION_MAP: [level 0, level 1])."""
    return {"techniques": TECHNIQUES, "elements": list(ELEMENTS), "tension_map": {k: list(v) for k, v in TENSION_MAP.items()}}


@tool()
def build_tension(clip: str, resolve_at: Pos, build_bars: Len = 8.0, elements: list[str] | None = None, peak: float = 0.9,
                  gap_beats: float = 0.5, impact: bool = True, reverb_throw: bool = True, respect_existing_gap: bool = True,
                  tension: list[list[float]] | None = None) -> dict:
    """Build tension for `build_bars` bars and release it on the downbeat `resolve_at` (a bar, "cue:drop" or "section:drop"): rising high-pass,
    bass kill and volume swell, a riser ending at the gap, a snare roll and a shrinking loop roll in the last bars, then `gap_beats` of
    silence, a reverb throw and an impact on the downbeat, with every lane back to neutral there. elements picks from list_techniques
    (default all). peak is the level 0..1 reached at the end of the default s-curve; tension=[[bar, level], ...] (bars of this clip) replaces
    the curve. If the track already has a gap before that section and respect_existing_gap is true, gap_beats becomes 0. The result is
    plain events tagged with a group: undo it with remove_technique."""
    with LOCK:
        c = clip_of(clip)
        at = at_of(c, resolve_at, "resolve_at")
        tn = None if tension is None else [[at_of(c, b) * BAR, float(v)] for b, v in tension]
        found = existing_gap(lib().analyze(lib().effective(c.track)["id"])["structure"]["sections"], c.src_start_beat / BAR + at)
        keep = bool(respect_existing_gap and found)
        gap = 0.0 if keep else gap_beats
        r = build_and_release(c, at, new_id("tn")[3:], len_of(build_bars, "build_bars"), elements, peak, gap, impact, reverb_throw, tn)
        save()
    return {"clip": clip, **r, "existing_gap_beats": found,
            "note": f"the track already has a gap of {found:g} beats before this section, so none was added" if keep else None}


@tool()
def remove_technique(clip: str, group: str) -> dict:
    """Undo a build_tension (or plan technique): delete every fx, loop and trigger tagged `group` (tension:<id>, shown by build_tension and
    export_plan) and reset the filter, bass EQ and volume lanes between the first and last of those events."""
    with LOCK:
        c = clip_of(clip)
        r = remove_group(c, group)
        save()
    return r
