from __future__ import annotations

from pathlib import Path

from .. import candidates, engine, mixing
from .. import plan as planlib
from ..core.units import BAR
from ..model.clip import Clip
from ..model.events import new_id
from ..model.setmodel import SetModel

FALLBACK = ("bass_swap", 16)


def pick_transition(lib, a: str, b: str, goals) -> tuple[str, int]:
    try:
        c = candidates.suggest(lib, a, b, goals, 1, 1)["candidates"]
        return (c[0]["style"], int(c[0]["bars"])) if c else FALLBACK
    except (ValueError, KeyError):
        return FALLBACK


def link(lib, model: SetModel, out: Clip, inc: Clip, style: str, bars: int) -> dict:
    room = int(min(out.length_beats, inc.length_beats) // BAR // 2)
    n = max(min(bars, room), 1)
    for st, nb in ((style, n), ("filter_fade", min(8, room)), ("drop_swap", min(4, room))):
        try:
            r = mixing.apply_template(model, lib.effective, out, inc, st, None, float(max(nb, 1)), "native", True, 0.0)
            mixing.clamp_to_clip(out, inc)
            return {"style": st, "bars": nb, **{k: r[k] for k in ("incoming_enters_at_bar_of_outgoing",) if k in r}}
        except ValueError:
            continue
    raise ValueError(f"no transition fits between {out.id} and {inc.id}")


def assemble(lib, name: str, ids: list[str], trims: list[dict], goals, styles: list[tuple[str, int]]) -> tuple[SetModel, list[dict]]:
    model = SetModel(name)
    links = []
    for tid, t in zip(ids, trims):
        c = Clip(new_id("c"), tid, t["start_bar"] * BAR, t["bars"] * BAR, label=Path(lib.effective(tid)["file"]).stem)
        if model.clips:
            c.start = {"anchor": model.clips[-1].id, "anchor_beat": 0.0}
        model.clips.append(c)
    for k in range(1, len(model.clips)):
        links.append(link(lib, model, model.clips[k - 1], model.clips[k], *styles[k - 1]))
    return model, links


def total_sec(lib, model: SetModel) -> float:
    rows = engine.timeline(model, lib.effective)
    return max(r["end_sec"] for r in rows) - min(r["start_sec"] for r in rows)


def to_plan(lib, model: SetModel) -> tuple[dict, list[dict]]:
    p = planlib.export_plan(model)
    return p, planlib.validate_plan(p, lib)
