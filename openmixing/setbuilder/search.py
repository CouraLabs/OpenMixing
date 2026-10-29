from __future__ import annotations

import math
import random

from .cost import Scorer

WIDTH = 64


def beam(sc: Scorer, width: int = WIDTH) -> tuple[list[tuple[float, tuple]], tuple]:
    n, items, c = len(sc.items), sc.items, sc.cons
    must = {sc.idx[t] for t in c.must_include if t in sc.idx} | ({sc.end} if sc.end is not None else set())
    maxlen = min(n, c.max_tracks or n)
    states, done, nearest = [(0.0, (), 0.0)], {}, ()
    for step in range(maxlen):
        nxt: dict = {}
        for cost, seq, el in states:
            closing = sc.end is not None and el >= sc.target - items[sc.end]["dur"]
            for j in ([sc.end] if closing else [k for k in range(n) if k != sc.end]):
                if not sc.allowed(seq, j):
                    continue
                s2, e2 = seq + (j,), el + items[j]["dur"]
                c2 = cost + sc.step(seq[-1] if seq else None, j, el)["cost"]
                full = must <= set(s2) and (j == sc.end if sc.end is not None else e2 >= sc.target or step == maxlen - 1)
                if full:
                    done[s2] = c2
                else:
                    key = (frozenset(s2), j)
                    if key not in nxt or c2 < nxt[key][0]:
                        nxt[key] = (c2, s2, e2)
        if not nxt:
            break
        states = sorted(nxt.values(), key=lambda s: (s[0] - 50 * len(must & set(s[1])), s[1]))[:width]
        nearest = states[0][1]
    return sorted(((v, k) for k, v in done.items())), nearest


def moves(seq: tuple):
    n = len(seq)
    for i in range(n):
        for j in range(n):
            if i < j:
                s = list(seq)
                s[i], s[j] = s[j], s[i]
                yield tuple(s)
                yield seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
            if i != j:
                s = list(seq)
                s.insert(j, s.pop(i))
                yield tuple(s)


def improve(sc: Scorer, seq: tuple, effort: str = "normal", seed: int = 0) -> tuple:
    best, best_c = seq, sc.total(seq)
    for _ in range(20):
        found = min(((sc.total(m), m) for m in moves(best) if not sc.violations(m)), default=None)
        if not found or found[0] >= best_c - 1e-9:
            break
        best_c, best = found
    if effort != "high" or len(best) < 3:
        return best
    rng, cur, cur_c, iters = random.Random(seed), best, best_c, 300
    for k in range(iters):
        m = rng.choice(list(moves(cur)))
        if sc.violations(m):
            continue
        mc, temp = sc.total(m), 2.0 * (1 - k / iters) + 0.05
        if mc < cur_c or rng.random() < math.exp(-(mc - cur_c) / temp):
            cur, cur_c = m, mc
            if mc < best_c:
                best, best_c = m, mc
    return best
