from __future__ import annotations

from collections import Counter

from .. import candidates
from ..mixing import audit
from . import arcs as arclib
from . import assemble as asm
from .constraints import parse
from .cost import Scorer
from .diagnose import infeasible
from .pool import LEAD_BARS, build_items
from .search import beam, improve
from .trim import trim


def distinct(done: list, n: int = 3) -> list[dict]:
    return [{"cost": round(c, 3), "seq": s} for c, s in done[:n]]


def arc_trace(sc: Scorer, order: tuple, trims: list[dict]) -> list[dict]:
    el, rows = 0.0, []
    for j, t in zip(order, trims):
        it = sc.items[j]
        f = min((el + it["dur"] / 2) / sc.target, 1.0)
        rows.append({"track": it["file"], "t_frac": round(f, 3), "energy": round(t["energy"], 2), "bpm": round(it["bpm"], 2),
                     "target_energy": round(arclib.at(sc.arcs["energy"], f), 2) if sc.arcs["energy"] else None,
                     "target_bpm": round(arclib.at(sc.arcs["tempo"], f), 2) if sc.arcs["tempo"] else None})
        el += it["dur"]
    return rows


def build(lib, pool, duration_min: float, energy_arc, tempo_arc, key_arc, tempo_follows_energy: bool, key_follows_energy: bool,
          constraints: dict | None, effort: str, seed: int, weights: dict | None, goals, include_plan: bool, name: str = "built_set") -> dict:
    cons = parse(constraints, lambda r: lib.resolve(r)["id"])
    facts: dict = {}
    fact = lambda tid: facts.setdefault(tid, candidates.track_facts(lib, tid))
    items, warns = build_items(lib, pool, cons, fact)
    if not items:
        return {"infeasible": ["pool"], "hint": "no analyzed tracks left after filtering"}
    target = duration_min * 60.0
    base = items[[i["id"] for i in items].index(cons.start_with)]["num"] if cons.start_with else Counter(i["num"] for i in items).most_common(1)[0][0]
    ar = arclib.make(items, energy_arc, tempo_arc, key_arc, tempo_follows_energy, key_follows_energy, base)
    order, trims, alts, energies = (), [], [], {}
    for _ in range(2):
        sc = Scorer([{**i, "energy": energies.get(i["id"], i["energy"])} for i in items], cons, ar, weights, target)
        done, nearest = beam(sc)
        if not done:
            return infeasible(sc, cons, nearest)
        new = improve(sc, done[0][1], effort, seed)
        alts = distinct(done)
        trims = trim([fact(sc.items[j]["id"]) for j in new], [sc.items[j]["bpm"] for j in new], target + 0.0, cons.min_play_bars)
        if new == order:
            break
        order, energies = new, {sc.items[j]["id"]: t["energy"] for j, t in zip(new, trims)}
    ids = [sc.items[j]["id"] for j in order]
    styles = [asm.pick_transition(lib, a, b, goals) for a, b in zip(ids, ids[1:])]
    extra = sum(min(b, 16) * 240.0 / sc.items[j]["bpm"] for (_, b), j in zip(styles, order[1:]))
    goal = target + extra
    for _ in range(3):
        trims = trim([fact(t) for t in ids], [sc.items[j]["bpm"] for j in order], goal, cons.min_play_bars)
        model, links = asm.assemble(lib, name, ids, trims, goals, styles)
        got = asm.total_sec(lib, model)
        if abs(got - target) <= target * cons.duration_tolerance_percent / 100 / 2:
            break
        goal += target - got
    plan, errs = asm.to_plan(lib, model)
    rep = audit(model, lib)
    if abs(got - target) > target * cons.duration_tolerance_percent / 100:
        warns.append(f"duration {got / 60:.1f} min is outside the {cons.duration_tolerance_percent:g}% tolerance of {duration_min:g} min")
    if cons.duration_max is not None and got > cons.duration_max * 60:
        warns.append(f"duration {got / 60:.1f} min exceeds duration_max {cons.duration_max:g}")
    rows = [{"id": sc.items[j]["id"], "file": sc.items[j]["file"], "camelot": sc.items[j]["camelot"], "src_start_bar": t["start_bar"],
             "bars": t["bars"], "transition": links[k - 1] if k else None} for k, (j, t) in enumerate(zip(order, trims))]
    alts = [{"cost": a["cost"], "order": [sc.items[j]["file"] for j in a["seq"]]} for a in alts]
    issues = rep["issues"]
    out = {"infeasible": None, "name": name, "duration_sec": round(got, 1), "target_sec": target, "n_tracks": len(ids), "effort": effort,
           "seed": seed, "order": rows, "arc": arc_trace(sc, order, trims), "cost": sc.breakdown(order), "alternatives": alts,
           "audit": {"errors": [i for i in issues if i["severity"] == "error"], "warns": sum(i["severity"] == "warn" for i in issues)},
           "plan_errors": errs, "warnings": warns, "lead_bars": LEAD_BARS}
    if include_plan:
        out["plan"] = plan
    return out
