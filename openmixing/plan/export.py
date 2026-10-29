from __future__ import annotations

from ..core.units import BAR, BEAT_LABELS
from ..model.lane import STEM_LANES
from ..model.setmodel import SetModel
from .exportfx import export_ducks, export_fx, export_trigger
from .schema import BANDS, PLAN_BASE_VERSION, PLAN_VERSION


def _r(x: float) -> float:
    return float(x)


def export_plan(model: SetModel) -> dict:
    clips = []
    for c in model.clips:
        o: dict = {"id": c.id, "track": c.track, "label": c.label, "src_start": _r(c.src_start_beat / BAR), "bars": _r(c.length_beats / BAR)}
        st = c.start
        if "anchor" in st:
            o["start"] = {"over": st["anchor"], "at_bar": _r(st["anchor_beat"] / BAR)}
        else:
            o["start"] = {"at_sec": _r(st.get("at_sec", 0.0))}
        if st.get("offset_ms"):
            o["start"]["offset_ms"] = st["offset_ms"]
        for k, default in (("gain_db", 0.0), ("semitones", 0.0), ("key_lock", True), ("stems", False)):
            if getattr(c, k) != default:
                o[k] = getattr(c, k)
        env = lambda lane: {"points": [[_r(b / BAR), _r(v)] for b, v in lane.points]}
        if c.tempo.points:
            o["tempo"] = env(c.tempo)
        if c.pitch.points:
            o["pitch"] = env(c.pitch)
        eq = {b: env(c.lanes[f"eq_{b}"]) for b in BANDS if c.lanes[f"eq_{b}"].points}
        if eq:
            o["eq"] = eq
        for k in ("filter", "volume"):
            if c.lanes[k].points:
                o[k] = env(c.lanes[k])
        sl = {s: env(c.lanes[ln]) for s, ln in STEM_LANES.items() if c.lanes[ln].points}
        if sl:
            o["stem_levels"] = sl
        if c.fx:
            o["fx"] = [export_fx(f) for f in c.fx]
        if c.loops:
            o["loops"] = [{"id": l.id, "at": _r(l.beat / BAR), "size_beats": BEAT_LABELS[l.size_beats], "bars": _r(l.length_beats / BAR),
                           "roll": l.roll, **({"group": l.group} if l.group else {})} for l in c.loops]
        if c.rate:
            o["rate"] = [{"id": r.id, "kind": r.kind, "at": _r(r.beat / BAR), "bars": _r(r.duration_beats / BAR), "points": r.curve, "slip": r.slip}
                         for r in c.rate]
        if c.triggers:
            o["triggers"] = [export_trigger(t) for t in c.triggers]
        clips.append(o)
    new = model.ducks or model.sampler or any(c.rate or c.triggers or any(f.chain or f.wet_curve for f in c.fx) for c in model.clips)
    version = PLAN_VERSION if new else PLAN_BASE_VERSION
    out = {"version": version, "name": model.name, "notes": model.notes, "clips": clips}
    if model.sampler:
        out["sampler"] = {k: v.to_dict() for k, v in model.sampler.items()}
    if model.ducks:
        out["ducks"] = export_ducks(model.ducks)
    if model.intent:
        out["intent"] = model.intent
    return out
