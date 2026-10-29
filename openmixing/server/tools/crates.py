from __future__ import annotations

from typing import Literal

from ... import engine
from ...library.store import SORTS, select
from ..app import tool
from ..state import LOCK, cur, info, lib


def page(rows: list[dict], limit: int, offset: int = 0) -> dict:
    return {"total": len(rows), "offset": offset, "tracks": rows[offset:offset + limit]}


def brief(tid: str) -> dict:
    t = next((x for x in lib().scan() if x["id"] == tid), None)
    d = lib().cached(t) if t else None
    return {"id": tid, "file": t["file"] if t else None, "bpm": round(d["tempo"]["bpm"], 2) if d else None,
            "camelot": d["key"]["camelot"] if d else None}


@tool()
def create_collection(name: str, notes: str = "") -> dict:
    """Create a named collection (a hand-picked list of tracks) or update its notes. Stored in .openmixing/collections.json by track id, so
    renaming or moving files keeps it."""
    with LOCK:
        return {"name": name, **lib().store.create(name, notes)}


@tool()
def collection_add(name: str, tracks: list[str]) -> dict:
    """Add tracks (ids, file names or unique parts of the name) to a collection."""
    with LOCK:
        c = lib().store.change(name, [lib().resolve(t)["id"] for t in tracks], [])
        return {"name": name, "count": len(c["tracks"]), "notes": c["notes"]}


@tool()
def collection_remove(name: str, tracks: list[str]) -> dict:
    """Remove tracks from a collection."""
    with LOCK:
        c = lib().store.change(name, [], [lib().resolve(t)["id"] for t in tracks])
        return {"name": name, "count": len(c["tracks"]), "notes": c["notes"]}


@tool()
def list_collections(name: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    """List collections with their size and notes; with `name`, list that collection's tracks (id, file, bpm, camelot), paginated by `limit`."""
    if name:
        return {"name": name, **page([brief(t) for t in lib().store.collection(name)["tracks"]], limit, offset)}
    cs = lib().store.collections()
    return {"total": len(cs), "collections": [{"name": n, "count": len(c["tracks"]), "notes": c["notes"]} for n, c in sorted(cs.items())]}


@tool()
def save_crate(name: str, rules: dict, sort: Literal[*SORTS] = "name") -> dict:
    """Save a smart crate: a rule set evaluated on demand against the sidecar data (analysis with overrides, set_tags rating / color / tags /
    notes / artist / title / genre, play history). Not the tags embedded in the files. Rules: {"all": [r, ...]} or {"any": [...]} (nestable),
    r = {"field", "op", "value"}. Fields: bpm, energy, duration, rating, play_count (eq ne between[lo,hi] in gte lte), bpm also
    compatible_with {bpm, percent=6}; camelot, color, artist, title, genre, notes (eq ne in contains), camelot also compatible_with {camelot,
    max_distance=1}; tag (contains / in / eq); has_stems (eq ne); last_played (within_days, gte, lte as epoch seconds). Unknown fields or ops
    are refused with the JSON path of the offending rule."""
    with LOCK:
        return {"name": name, **lib().store.save_crate(name, rules, sort)}


@tool()
def get_crate(name: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    """Evaluate a smart crate now and return its tracks (id, file, bpm, camelot, energy, play_count), paginated by `limit` with `total`.
    Without `name`, list the saved crates."""
    if name is None:
        return {"crates": lib().store.crates()}
    c = lib().store.crate(name)
    return {"name": name, "sort": c["sort"], **page(select(lib(), c["rules"], c["sort"]), limit, offset)}


@tool()
def tag_track(track: str, add: list[str] | None = None, remove: list[str] | None = None) -> dict:
    """Add and/or remove labels on a track's `tags` list (lowercased, trimmed) without touching its other tags."""
    with LOCK:
        t = lib().resolve(track)
        cur_tags = lib().user_tags(t["id"]).get("tags", [])
        rm = {x.strip().lower() for x in remove or []}
        new = [x for x in [*cur_tags, *(a.strip().lower() for a in add or [])] if x not in rm]
        return {"track": t["file"], "tags": lib().set_user_tags(t["id"], {"tags": new})["tags"]["tags"]}


@tool()
def record_play(set: str | None = None, tag: Literal["played", "rendered"] = "played") -> dict:
    """Append the tracks of a set (default: the current one) with their start / end seconds to the play history
    (.openmixing/history.jsonl). Only calls to this tool write history; only `played` entries count for get_history, play_count and last_played."""
    with LOCK:
        m = lib().load_set(set) if set else cur()
        rows = [{"id": info(r["track"])["id"], "start_sec": r["start_sec"], "end_sec": r["end_sec"]} for r in engine.timeline(m, info)]
        render = next((f"{lib().safe_name(m.name)}.{x}" for x in ("mp3", "flac", "wav") if lib().render_path(m.name, x).exists()), None)
        rec = lib().store.history.append(m.name, rows, tag, render)
        return {"set": m.name, "tag": tag, "ts": rec["ts"], "tracks": len(rows)}


@tool()
def get_history(track: str | None = None, since_days: float | None = None, limit: int = 50, offset: int = 0) -> dict:
    """Play history (newest first): when each track was played, in which set, optionally for one `track` and/or the last `since_days` days."""
    tid = lib().resolve(track)["id"] if track else None
    files = {t["id"]: t["file"] for t in lib().scan()}
    rows = sorted(lib().store.history.plays(tid, since_days), key=lambda r: -r["ts"])
    rows = [{**r, "file": files.get(r["id"])} for r in rows]
    return {"total": len(rows), "offset": offset, "plays": rows[offset:offset + limit]}
