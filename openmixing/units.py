"""Flexible positions: bars, beats, seconds, cue names or section names, all resolved to bars of one track."""
from __future__ import annotations

import re

from .errors import OpenMixingError

BEATS_PER_BAR = 4.0
_NUM = r"-?\d+(?:\.\d+)?"


def to_bars(value, e: dict, cues: dict[str, float] | None = None, sections: list[dict] | None = None,
            field: str = "position") -> float:
    """Resolve a position to bars (0 = first downbeat) of the track described by effective-info `e`.

    Accepted: 16, "16", "16 bars", "bar 16", "64 beats", "12.5s", "cue:drop", "cue:my cue+2", "section:breakdown",
    "section:drop+4" (offset in bars), "section:drop-1". A bare name is tried as a cue, then as a section label."""
    cues, sections = cues or {}, sections or []
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    low = s.lower()
    if m := re.fullmatch(rf"({_NUM})", low) or re.fullmatch(rf"(?:bars?\s+)?({_NUM})(?:\s*bars?)?", low):
        return float(m.group(1))
    if m := re.fullmatch(rf"({_NUM})\s*beats?", low) or re.fullmatch(rf"beats?\s+({_NUM})", low):
        return float(m.group(1)) / BEATS_PER_BAR
    if m := re.fullmatch(rf"({_NUM})\s*(?:s|sec|secs|seconds)", low):
        return (float(m.group(1)) - e["first_downbeat"]) * e["bpm"] / (60.0 * BEATS_PER_BAR)
    kind, _, name = low.partition(":")
    if not name:
        kind, name = "", low
    off = 0.0
    if m := re.fullmatch(rf"(.+?)\s*([+-]\s*{_NUM})\s*(?:bars?)?", name):
        name, off = m.group(1).strip(), float(m.group(2).replace(" ", ""))
    if kind in ("", "cue") and name in {k.lower(): k for k in cues}:
        return cues[{k.lower(): k for k in cues}[name]] + off
    if kind in ("", "section"):
        hits = [x for x in sections if x["label"].lower() == name]
        if hits:
            return float(hits[0]["start_bar"]) + off
    known = {"cues": sorted(cues), "sections": sorted({x["label"] for x in sections})}
    raise OpenMixingError("invalid_position", f"cannot resolve {value!r} to a position on {e.get('file', 'the track')}", field,
                        f"use bars (16), '64 beats', '12.5s', 'cue:<name>' or 'section:<label>'; known: {known}")


_OUTPUT_FORM = re.compile(rf"(?:bars?\s+)?{_NUM}(?:\s*bars?)?|{_NUM}\s*beats?|beats?\s+{_NUM}")


def is_output_unit(value) -> bool:
    """True for plain bars / beats, which are counted from the clip's own start; cue, section and seconds are source-track units."""
    return isinstance(value, (int, float)) or bool(_OUTPUT_FORM.fullmatch(str(value).strip().lower()))


def out_bars(value, src_start_bar: float, to_src, field: str = "position") -> float:
    """A position inside a clip's playback, in output bars (0 = the clip's first bar).

    Bars ("8", 8, "8 bars") and beats ("32 beats") are counted from the clip start. Cue, section and seconds positions name a
    place in the source track: `to_src(value)` turns them into source bars and the clip's start is subtracted, so
    "cue:drop" lands where that cue plays (loops and tempo changes before it are not taken into account)."""
    if is_output_unit(value):
        return to_bars(value, {"bpm": 120.0, "first_downbeat": 0.0}, None, None, field)
    if to_src is None:
        raise OpenMixingError("invalid_position", f"{value!r} names a place in the source track but there is no track to resolve it on", field,
                            "use bars or 'N beats' here")
    bar = to_src(value) - src_start_bar
    if bar < -1e-9:
        raise OpenMixingError("out_of_range", f"{value!r} is {-bar:g} bars before the clip starts playing (src_start_bar {src_start_bar:g})", field,
                            "move the clip's src_start earlier or use a later position")
    return max(bar, 0.0)


def duration_bars(value, field: str = "bars") -> float:
    """A length in bars: a number, "8 bars" or "32 beats"."""
    if not is_output_unit(value):
        raise OpenMixingError("invalid_position", f"{value!r} is not a length", field, "use a number of bars or 'N beats'")
    return to_bars(value, {"bpm": 120.0, "first_downbeat": 0.0}, None, None, field)
