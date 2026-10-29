from __future__ import annotations

import math
import re

from .errors import OpenMixingError

BAR = 4.0
SR = 48_000
BEAT_DIVISIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
BEAT_LABELS = {1 / 16: "1/16", 1 / 8: "1/8", 1 / 4: "1/4", 1 / 2: "1/2", 1.0: "1", 2.0: "2", 4.0: "4", 8.0: "8",
               16.0: "16", 32.0: "32"}


def parse_beats(value) -> float:
    try:
        if isinstance(value, str):
            s = value.strip()
            if "/" in s:
                n, d = s.split("/", 1)
                v = float(n) / float(d)
            else:
                v = float(s)
        else:
            v = float(value)
    except ZeroDivisionError as e:
        raise ValueError(f"invalid beat length {value!r}") from e
    for b in BEAT_DIVISIONS:
        if abs(b - v) < 1e-9:
            return b
    raise ValueError(f"beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {value!r}")


_NUM = r"-?\d+(?:\.\d+)?"


def to_bars(value, e: dict, cues: dict[str, float] | None = None, sections: list[dict] | None = None,
            field: str = "position") -> float:
    cues, sections = cues or {}, sections or []
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise OpenMixingError("invalid_position", f"{value!r} is not a finite position", field, "use a finite number of bars")
        return float(value)
    s = str(value).strip()
    low = s.lower()
    if m := re.fullmatch(rf"({_NUM})", low) or re.fullmatch(rf"(?:bars?\s+)?({_NUM})(?:\s*bars?)?", low):
        return float(m.group(1))
    if m := re.fullmatch(rf"({_NUM})\s*beats?", low) or re.fullmatch(rf"beats?\s+({_NUM})", low):
        return float(m.group(1)) / BAR
    if m := re.fullmatch(rf"({_NUM})\s*(?:s|sec|secs|seconds)", low):
        return (float(m.group(1)) - e["first_downbeat"]) * e["bpm"] / (60.0 * BAR)
    kind, _, name = low.partition(":")
    if not name:
        kind, name = "", low
    off = 0.0
    if m := re.fullmatch(rf"(.+?)\s*([+-]\s*{_NUM})\s*(?:bars?)?", name):
        name, off = m.group(1).strip(), float(m.group(2).replace(" ", ""))
    by_name = {k.lower(): k for k in cues}
    if kind in ("", "cue") and name in by_name:
        return cues[by_name[name]] + off
    if kind in ("", "section"):
        hits = [x for x in sections if x["label"].lower() == name]
        if hits:
            return float(hits[0]["start_bar"]) + off
    known = {"cues": sorted(cues), "sections": sorted({x["label"] for x in sections})}
    raise OpenMixingError("invalid_position", f"cannot resolve {value!r} to a position on {e.get('file', 'the track')}", field,
                        f"use bars (16), '64 beats', '12.5s', 'cue:<name>' or 'section:<label>'; known: {known}")


_OUTPUT_FORM = re.compile(rf"(?:bars?\s+)?{_NUM}(?:\s*bars?)?|{_NUM}\s*beats?|beats?\s+{_NUM}")


def is_output_unit(value) -> bool:
    return isinstance(value, (int, float)) or bool(_OUTPUT_FORM.fullmatch(str(value).strip().lower()))


def out_bars(value, src_start_bar: float, to_src, field: str = "position") -> float:
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
    if not is_output_unit(value):
        raise OpenMixingError("invalid_position", f"{value!r} is not a length", field, "use a number of bars or 'N beats'")
    return to_bars(value, {"bpm": 120.0, "first_downbeat": 0.0}, None, None, field)
