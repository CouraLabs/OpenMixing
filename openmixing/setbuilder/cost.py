from __future__ import annotations

from ..mixing import compat
from . import arcs as arclib
from .constraints import Constraints

WEIGHTS = {"transition": 1.0, "energy": 3.0, "tempo": 0.3, "key": 1.0}


def circ(a: float, b: float) -> float:
    d = abs(a - b) % 12
    return min(d, 12 - d)


class Scorer:
    def __init__(self, items: list[dict], cons: Constraints, arcs: dict, weights: dict | None, target: float):
        self.items, self.cons, self.arcs, self.target = items, cons, arcs, max(target, 1.0)
        self.w = {**WEIGHTS, **(weights or {})}
        self.idx = {it["id"]: i for i, it in enumerate(items)}
        self.fixed = {p: self.idx[t] for p, t in cons.fixed_positions.items() if t in self.idx}
        if cons.start_with in self.idx:
            self.fixed[0] = self.idx[cons.start_with]
        self.end = self.idx.get(cons.end_with)
        self.banned = {(self.idx[a], self.idx[b]) for a, b in cons.must_not_follow if a in self.idx and b in self.idx}
        self._pair: dict = {}

    def pair(self, i: int, j: int) -> dict:
        if (i, j) not in self._pair:
            self._pair[i, j] = compat(self.items[i], self.items[j])
        return self._pair[i, j]

    def pair_ok(self, i: int, j: int) -> bool:
        c, p = self.cons, self.pair(i, j)
        return ((i, j) not in self.banned and (c.max_bpm_jump_percent is None or abs(p["tempo"]["stretch_percent"]) <= c.max_bpm_jump_percent)
                and (c.max_key_distance is None or p["key"]["distance"] <= c.max_key_distance))

    def artist_ok(self, seq: tuple, j: int) -> bool:
        g, a = self.cons.min_gap_between_artist_repeats, self.items[j]["artist"]
        return not (g and a and any(self.items[k]["artist"] == a for k in seq[-g:]))

    def allowed(self, seq: tuple, j: int) -> bool:
        pos = len(seq)
        pinned = set(self.fixed.values())
        if j in seq or (self.fixed.get(pos, j) != j) or (j in pinned and self.fixed.get(pos) != j):
            return False
        return (not seq or self.pair_ok(seq[-1], j)) and self.artist_ok(seq, j)

    def violations(self, seq: tuple) -> list[str]:
        c, out = self.cons, []
        if any(not self.allowed(seq[:k], j) for k, j in enumerate(seq)):
            out.append("pair_or_position_rules")
        if any(self.idx.get(t) not in seq for t in c.must_include):
            out.append("must_include")
        if self.end is not None and seq[-1:] != (self.end,):
            out.append("end_with")
        if c.max_tracks and len(seq) > c.max_tracks:
            out.append("max_tracks")
        return out

    def step(self, prev: int | None, j: int, elapsed: float) -> dict:
        it, ar = self.items[j], self.arcs
        f = (elapsed + it["dur"] / 2) / self.target
        trans = 0.0 if prev is None else (100 - self.pair(prev, j)["score"]) / 10
        en = abs(it["energy"] - arclib.at(ar["energy"], f)) if ar["energy"] else 0.0
        tp = abs(it["bpm"] - arclib.at(ar["tempo"], f)) if ar["tempo"] else 0.0
        ky = circ(it["num"], arclib.at(ar["key"], f)) if ar["key"] else 0.0
        parts = {"transition": trans, "energy": en, "tempo": tp, "key": ky}
        return {"track": it["file"], "t_frac": round(min(f, 1.0), 3), "raw": {k: round(v, 3) for k, v in parts.items()},
                "cost": sum(self.w[k] * v for k, v in parts.items())}

    def breakdown(self, seq: tuple) -> dict:
        steps, el = [], 0.0
        for k, j in enumerate(seq):
            steps.append(self.step(seq[k - 1] if k else None, j, el))
            el += self.items[j]["dur"]
        keys = ("transition", "energy", "tempo", "key")
        return {"steps": steps, "total": round(sum(s["cost"] for s in steps), 3),
                "by_term": {k: round(sum(self.w[k] * s["raw"][k] for s in steps), 3) for k in keys}}

    def total(self, seq: tuple) -> float:
        return self.breakdown(seq)["total"]
