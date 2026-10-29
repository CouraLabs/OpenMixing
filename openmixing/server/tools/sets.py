from __future__ import annotations

from ... import mixing
from ...model.setmodel import SetModel
from ..app import tool
from ..state import LOCK, STATE, cur, lib, save, span
from ..versions import versions
from ..views import clip_row, describe_events, jsonable


@tool()
def list_sets() -> dict:
    """Saved sets in this folder and the current one."""
    return {"sets": lib().list_sets(), "current": cur().name}


@tool()
def new_set(name: str, notes: str = "") -> dict:
    """Create (or reset) a set and make it current."""
    with LOCK:
        STATE["set"] = SetModel(name, notes=notes)
        save()
    return {"current": name}


@tool()
def open_set(name: str) -> dict:
    """Make a saved set current."""
    with LOCK:
        STATE["set"] = lib().load_set(name)
        save()
    return {"current": name, "clips": len(cur().clips)}


@tool()
def get_set(detail: bool = False) -> dict:
    """The current set: clips with resolved times, plus a readable event list per clip (detail=True adds raw automation points)."""
    m = cur()
    clips = []
    for c in sorted(m.clips, key=lambda c: span(c)[0]):
        row = clip_row(c)
        row["events"] = describe_events(c)
        if detail:
            row["raw"] = c.to_dict()
        clips.append(row)
    return jsonable({"name": m.name, "notes": m.notes, "duration_sec": max((r["end_sec"] for r in clips), default=0), "clips": clips,
                     "ducks": [f"{d.id}: {d.target}{':' + d.stem if d.stem else ''} ducked {d.depth_db:g} dB by {d.key} from bar "
                               f"{d.beat / 4:g} for {d.duration_beats / 4:g} bars" for d in m.ducks], "versions": versions()})


@tool()
def audit_set() -> dict:
    """Lint the set like a DJ coach: clips running past their track, off-phrase entries, bass clashes (both low bands open), key clashes,
    big tempo stretches, unstable grids, silent gaps. Fix every 'error' and 'warn' before rendering."""
    return jsonable(mixing.audit(cur(), lib()))
