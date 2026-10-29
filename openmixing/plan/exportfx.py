from __future__ import annotations

from ..core.units import BAR, BEAT_LABELS


def _step(fx: str, beats: float, wet: float, params: dict) -> dict:
    return {"fx": fx, "beats": BEAT_LABELS[beats], "wet": wet, **({"params": params} if params else {})}


def export_fx(f) -> dict:
    o = {"id": f.id, "at": float(f.beat / BAR), "bars": float(f.duration_beats / BAR), "wet": f.wet}
    if f.chain:
        o["chain"] = [_step(s["fx"], s["beats"], s["wet"], s.get("params") or {}) for s in f.chain]
    else:
        o.update(_step(f.fx, f.beats, f.wet, f.params))
    if f.wet_curve:
        o["wet_points"] = [[b / BAR, w] for b, w in f.wet_curve]
    if f.stem:
        o["stem"] = f.stem
    if f.group:
        o["group"] = f.group
    return o


def export_ducks(ducks) -> list[dict]:
    return [{"id": d.id, "target": d.target, "key": d.key, "at": float(d.beat / BAR), "bars": float(d.duration_beats / BAR),
             "depth_db": d.depth_db, "threshold_db": d.threshold_db, "attack_ms": d.attack_ms, "release_ms": d.release_ms,
             **({"stem": d.stem} if d.stem else {})} for d in ducks]


def export_trigger(t) -> dict:
    return {"id": t.id, "kind": t.kind, "at": float(t.beat / BAR), "bars": float(t.duration_beats / BAR), "align": t.align, "gain_db": t.gain_db,
            **({"params": t.params} if t.params else {}), **({"slot": t.slot} if t.slot else {}),
            **({"group": t.group} if t.group else {})}
