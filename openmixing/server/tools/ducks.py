from __future__ import annotations

from typing import Literal

from ...core.errors import OpenMixingError
from ...core.units import BAR
from ...model.events import DuckEvent, new_id
from ...plan.ranges import RANGES
from ..app import Len, Pos, tool
from ..edit import without
from ..state import LOCK, at_of, clip_of, cur, len_of, save

Stem = Literal["vocals", "drums", "bass", "other"]


@tool()
def add_duck(target: str, key: str, at_bar: Pos, bars: Len, depth_db: float = 6.0, threshold_db: float = -30.0, attack_ms: float = 5.0,
             release_ms: float = 120.0, stem: Stem | None = None) -> dict:
    """Sidechain duck: lower `target` (a clip) whenever the key signal is above threshold_db, from `at_bar` (a bar of the target) for `bars`.
    key is "self:<stem>" (a stem of the target itself, e.g. "self:drums"; needs stem = the target stem to lower, e.g. "vocals", and
    stems=True) or another clip id (its whole rendered deck, e.g. the outgoing track's kicks pushing the incoming one down; stem
    optional). depth_db is the maximum reduction (0-40); attack_ms / release_ms smooth it. Cross-deck keys cost one extra render of the key
    clip over the ducked window."""
    with LOCK:
        t = clip_of(target)
        head, _, ks = key.partition(":")
        if head in ("self", target):
            if not ks or not stem or not t.stems:
                raise OpenMixingError("bad_key", "a self key is \"self:<stem>\", needs stem=<stem to lower> and a clip playing stems", "key",
                                      "call separate_stems, edit_clip(stems=True), then add_duck(key='self:drums', stem='vocals')")
            key = f"self:{ks}"
        elif ks or head not in {c.id for c in cur().clips}:
            raise OpenMixingError("bad_key", f"key {key!r} is not 'self:<stem>' or a clip id", "key", f"one of {[c.id for c in cur().clips]}")
        else:
            key = head
        if stem and not t.stems:
            raise OpenMixingError("stems_off", f"clip {target} plays the full mix", "stem", "call separate_stems, then edit_clip(stems=True)")
        v = dict(depth_db=depth_db, threshold_db=threshold_db, attack_ms=attack_ms, release_ms=release_ms)
        for k, x in v.items():
            lo, hi = RANGES[k]
            if not lo <= x <= hi:
                raise OpenMixingError("out_of_range", f"{k} {x} is outside [{lo}, {hi}]", k, f"use a value between {lo} and {hi}")
        e = DuckEvent(new_id("dk"), target, key, at_of(t, at_bar) * BAR, len_of(bars) * BAR, **v, stem=stem)
        cur().ducks.append(e)
        save()
    return {"duck_id": e.id, "target": target, "key": key}


@tool()
def remove_duck(duck_id: str) -> dict:
    """Remove a sidechain duck."""
    with LOCK:
        cur().ducks = without(cur().ducks, duck_id, "duck_id")
        save()
    return {"remaining_ducks": [d.id for d in cur().ducks]}
