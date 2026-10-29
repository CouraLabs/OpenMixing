from __future__ import annotations

import json

from ..core.units import BAR, BEAT_LABELS
from ..model.clip import Clip
from .state import info, span


def clip_row(c: Clip) -> dict:
    e = info(c.track)
    s, t = span(c)
    return {"clip": c.id, "label": c.label, "track": e["file"], "start_sec": round(s, 2), "end_sec": round(t, 2),
            "src_start_bar": c.src_start_beat / BAR, "bars": c.length_beats / BAR, "bpm_native": round(e["bpm"], 2),
            "camelot": e["camelot"], "stems": c.stems, "start": c.start}


def describe_events(c: Clip) -> list[str]:
    ev = []
    for name, lane in c.lanes.items():
        pts = lane.points
        for i, (b, v) in enumerate(pts):
            if i + 1 < len(pts) and abs(pts[i + 1][1] - v) < 1e-9 and (i == 0 or abs(pts[i - 1][1] - v) < 1e-9):
                continue
            if i == 0 and abs(v - lane.neutral) < 1e-9 and (len(pts) < 2 or pts[1][0] - b < 0.03):
                continue
            ev.append((b, f"{name} = {v:g}"))
    for i, (b, v) in enumerate(c.tempo.points):
        ev.append((b, f"tempo = {v:.2f} bpm"))
    for b, v in c.pitch.points:
        ev.append((b, f"pitch = {v:g} semitones"))
    groups: dict[str, list] = {}
    for e in (*c.fx, *c.loops, *c.triggers):
        if e.group:
            groups.setdefault(e.group, []).append(e)
    for g, es in groups.items():
        ev.append((min(getattr(e, "start", e.beat) for e in es), f"technique {g}: {len(es)} fx / loop / trigger events (remove_technique)"))
    for f in c.fx:
        if f.group:
            continue
        stem = f" on {f.stem} only" if f.stem else ""
        name = "chain " + " > ".join(x["fx"] for x in f.chain) if f.chain else f.fx
        what = f"fx {f.id}: {name}{stem} {BEAT_LABELS.get(f.beats, f.beats)} beats, wet {f.wet:g}, for {f.duration_beats / BAR:g} bars"
        ev.append((f.beat, f"{what} {f.params or ''}".strip()))
    for l in c.loops:
        if l.group:
            continue
        size, roll = BEAT_LABELS.get(l.size_beats, l.size_beats), "roll " if l.roll else ""
        ev.append((l.beat, f"loop {l.id}: {size}-beat {roll}for {l.length_beats / BAR:g} bars"))
    for r in c.rate:
        slip = "slip " if r.slip else ""
        ev.append((r.beat, f"rate {r.id}: {slip}{r.kind} for {r.duration_beats / BAR:g} bars, speed {r.curve[0][1]:g} -> {r.curve[-1][1]:g}"))
    for t in c.triggers:
        if t.group:
            continue
        ev.append((t.start, f"trigger {t.id}: {t.slot or t.kind} {'ending at bar ' + format(t.beat / BAR, 'g') + ', ' if t.align == 'end' else ''}"
                            f"for "
                            f"{t.duration_beats / BAR:g} bars, {t.gain_db:g} dB {t.params or ''}".strip()))
    return [f"bar {b / BAR:g}: {t}" for b, t in sorted(ev)]


def lane_name(band: str) -> str:
    return {"low": "eq_low", "mid": "eq_mid", "high": "eq_high"}[band]


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
