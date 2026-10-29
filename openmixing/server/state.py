from __future__ import annotations

import threading

from ..core.errors import OpenMixingError
from ..core.units import BAR, duration_bars
from ..jobs import JobManager
from ..library import Library
from ..model.clip import Clip
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap, resolve_starts

LOCK = threading.RLock()
STATE: dict = {"lib": None, "set": None, "ui": None}


def lib() -> Library:
    if STATE["lib"] is None:
        STATE["lib"] = Library()
    return STATE["lib"]


def jobs() -> JobManager:
    path = lib().state / "jobs.jsonl"
    if STATE.get("jobs") is None or STATE["jobs"].path != path:
        STATE["jobs"] = JobManager(path)
    return STATE["jobs"]


def cur() -> SetModel:
    if STATE["set"] is None:
        names = lib().list_sets()
        last = lib().state / "current_set"
        want = last.read_text().strip() if last.exists() else None
        if want in names:
            STATE["set"] = lib().load_set(want)
        else:
            STATE["set"] = SetModel("main")
            lib().save_set(STATE["set"])
    return STATE["set"]


def save() -> None:
    m = cur()
    lib().save_set(m)
    (lib().state / "current_set").write_text(m.name)


def clip_of(cid: str) -> Clip:
    return cur().clip(cid)


def info(track_id: str) -> dict:
    return lib().effective(track_id)


def span(c: Clip) -> tuple[float, float]:
    st = resolve_starts(cur(), lambda t: info(t)["bpm"])[c.id]
    return st, st + float(TempoMap(c, info(c.track)["bpm"]).beat_to_sec(c.length_beats))


def conv_beat(src: Clip, beat: float, dst: Clip) -> float:
    starts = resolve_starts(cur(), lambda t: info(t)["bpm"])
    t = starts[src.id] + float(TempoMap(src, info(src.track)["bpm"]).beat_to_sec(beat))
    return float(TempoMap(dst, info(dst.track)["bpm"]).sec_to_beat(t - starts[dst.id]))


def at_of(c: Clip, v, field: str = "at_bar") -> float:
    bars, n = lib().out_bars(c, v, field), c.length_beats / BAR
    if not 0 <= bars <= n:
        raise OpenMixingError("out_of_range", f"{field} {bars:g} is outside the clip (0 to {n:g} bars)", field, f"use a bar between 0 and {n:g}")
    return bars


def len_of(v, field: str = "bars") -> float:
    return duration_bars(v, field)
