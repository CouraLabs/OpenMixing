from __future__ import annotations

import time

from ..core.errors import OpenMixingError
from ..library.store import select
from .constraints import Constraints, pinned
from .trim import window_span

LEAD_BARS = 16


def refs(lib, pool, cons: Constraints) -> list[str]:
    if isinstance(pool, str):
        pool = [r["id"] for r in select(lib, lib.store.crate(pool)["rules"])]
    ids = list(pool) if pool else [t["id"] for t in lib.scan() if lib.cached(t)]
    if cons.crate:
        keep = {r["id"] for r in select(lib, lib.store.crate(cons.crate)["rules"])}
        ids = [i for i in ids if i in keep]
    return list(dict.fromkeys(lib.resolve(r)["id"] for r in ids))


def item(lib, f: dict, energy: float | None = None) -> dict:
    e, tid = f["e"], f["e"]["id"]
    d = f["d"]
    a, b = window_span(f)
    artist = lib.user_tags(tid).get("artist") or (d.get("tags") or {}).get("artist")
    dur = max(b - a - LEAD_BARS, 8) * 240.0 / e["bpm"]
    return {"id": tid, "file": e["file"], "bpm": e["bpm"], "camelot": e["camelot"], "num": int(e["camelot"][:-1]),
            "energy": float(e["energy"] if energy is None else energy), "dur": dur, "artist": artist, "stems": bool(e["stems"])}


def build_items(lib, pool, cons: Constraints, fact, energies: dict | None = None) -> tuple[list[dict], list[str]]:
    keep, warns = pinned(cons), []
    ids = [i for i in dict.fromkeys([*refs(lib, pool, cons), *pinned(cons)])]
    last = lib.store.history.stats() if cons.avoid_played_within_days is not None else {}
    floor = time.time() - (cons.avoid_played_within_days or 0) * 86400.0
    files, out = {t["id"]: t for t in lib.scan()}, []
    for tid in ids:
        if tid in cons.must_exclude or not lib.cached(files[tid]):
            if tid in keep:
                raise OpenMixingError("invalid_value", f"pinned track {tid} is excluded or not analyzed", "constraints",
                                      "analyze it / remove it from must_exclude")
            continue
        if tid not in keep and ((cons.only_with_stems and not lib.stem_paths(files[tid]))
                                or last.get(tid, {}).get("last_played", 0) >= floor > 0):
            continue
        out.append(item(lib, fact(tid), (energies or {}).get(tid)))
    if cons.only_with_stems and any(not i["stems"] for i in out):
        warns.append("pinned tracks without stems were kept")
    return out, warns
