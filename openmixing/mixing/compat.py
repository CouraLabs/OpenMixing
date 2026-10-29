from __future__ import annotations

from ..analysis.key import camelot_relation


def tempo_match(bpm_from: float, bpm_to: float) -> dict:
    best = min((bpm_to * f for f in (0.5, 1.0, 2.0)), key=lambda b: abs(b / bpm_from - 1))
    pct = (bpm_from / best - 1) * 100
    return {"bpm_from": round(bpm_from, 2), "bpm_to": round(bpm_to, 2), "ratio_used": round(best / bpm_to, 3),
            "stretch_percent": round(pct, 2),
            "verdict": "seamless" if abs(pct) <= 3 else "ok (use a tempo ramp)" if abs(pct) <= 8 else "hard (large stretch; prefer a cut / echo-out)"}


def compat(ea: dict, eb: dict) -> dict:
    dist, rel = camelot_relation(ea["camelot"], eb["camelot"])
    tm = tempo_match(ea["bpm"], eb["bpm"])
    score = 100 - min(abs(tm["stretch_percent"]) * 6, 45) - {0: 0, 1: 4, 2: 12}.get(dist, 30) - abs(ea["energy"] - eb["energy"]) * 2
    return {"score": round(max(score, 0), 1), "key": {"from": ea["camelot"], "to": eb["camelot"], "distance": dist, "relation": rel},
            "tempo": tm, "energy": {"from": ea["energy"], "to": eb["energy"], "delta": eb["energy"] - ea["energy"]}}


def order_tracks(effs: list[dict], strategy: str = "harmonic", start: str | None = None) -> list[dict]:
    pool = list(effs)
    if not pool:
        return []
    if start:
        cur = next(e for e in pool if e["id"] == start)
    elif strategy == "bpm_ramp":
        cur = min(pool, key=lambda e: e["bpm"])
    elif strategy == "energy_build":
        cur = min(pool, key=lambda e: e["energy"])
    else:
        cur = pool[0]
    path = [cur]
    pool.remove(cur)
    while pool:
        def cost(e):
            c = compat(path[-1], e)
            base = -c["score"]
            if strategy == "energy_build":
                base += 6 * max(path[-1]["energy"] - e["energy"], 0) - 2 * (e["energy"] - path[-1]["energy"])
            if strategy == "bpm_ramp":
                base += 3 * max(path[-1]["bpm"] - e["bpm"], 0)
            return base
        nxt = min(pool, key=cost)
        path.append(nxt)
        pool.remove(nxt)
    return path
