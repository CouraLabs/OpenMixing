from __future__ import annotations

from . import rules as ruleslib


def facts(lib, t: dict, stats: dict) -> dict:
    ut = lib.user_tags(t["id"])
    f = {"artist": ut.get("artist"), "title": ut.get("title"), "genre": ut.get("genre"), "notes": ut.get("notes"), "rating": ut.get("rating"),
         "color": ut.get("color"), "tag": ut.get("tags", []), "has_stems": bool(lib.stem_paths(t)), **stats.get(t["id"], {"play_count": 0})}
    if lib.cached(t):
        e = lib.effective(t["id"])
        f.update(bpm=e["bpm"], camelot=e["camelot"], energy=e["energy"], duration=e["duration"])
    return f


def sort_key(sort: str):
    return {"name": lambda r: r["file"].lower(), "bpm": lambda r: r.get("bpm") or 0, "energy": lambda r: r.get("energy") or 0,
            "rating": lambda r: -(r.get("rating") or 0), "duration": lambda r: r.get("duration") or 0,
            "play_count": lambda r: -r["play_count"], "last_played": lambda r: -(r.get("last_played") or 0)}[sort]


def select(lib, rules: dict, sort: str = "name") -> list[dict]:
    ruleslib.check(rules)
    stats = lib.store.history.stats()
    rows = []
    for t in lib.scan():
        f = facts(lib, t, stats)
        if ruleslib.match(rules, f):
            rows.append({"id": t["id"], "file": t["file"], "bpm": round(f["bpm"], 2) if "bpm" in f else None, "camelot": f.get("camelot"),
                         "energy": f.get("energy"), "duration": f.get("duration"), "rating": f.get("rating"), "play_count": f["play_count"],
                         "last_played": f.get("last_played")})
    return sorted(rows, key=sort_key(sort))
