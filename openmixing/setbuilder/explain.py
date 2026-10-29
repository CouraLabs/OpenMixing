from __future__ import annotations

from collections import Counter

from .. import candidates, engine
from ..core.units import BAR
from ..model.setmodel import SetModel
from . import arcs as arclib
from .constraints import parse
from .cost import Scorer
from .pool import item
from .trim import slice_energy


def explain(lib, model: SetModel, energy_arc, tempo_arc, key_arc, tempo_follows_energy: bool, key_follows_energy: bool,
            constraints: dict | None, weights: dict | None) -> dict:
    rows = sorted(engine.timeline(model, lib.effective), key=lambda r: r["start_sec"])
    clips = {c.id: c for c in model.clips}
    items = []
    for r in rows:
        c = clips[r["clip"]]
        a = c.src_start_beat / BAR
        f = candidates.track_facts(lib, r["track"])
        items.append({**item(lib, f, slice_energy(f, a, a + c.length_beats / BAR)), "dur": r["duration_sec"]})
    cons = parse(constraints, lambda x: lib.resolve(x)["id"])
    total = max(r["end_sec"] for r in rows) - min(r["start_sec"] for r in rows)
    base = Counter(i["num"] for i in items).most_common(1)[0][0]
    ar = arclib.make(items, energy_arc, tempo_arc, key_arc, tempo_follows_energy, key_follows_energy, base)
    sc = Scorer(items, cons, ar, weights, total)
    seq = tuple(range(len(items)))
    return {"set": model.name, "duration_sec": round(total, 1), "order": [i["file"] for i in items], "cost": sc.breakdown(seq),
            "violations": sc.violations(seq), "weights": sc.w}
