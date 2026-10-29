from __future__ import annotations

from ..core.units import BAR
from ..model.clip import Clip
from ..model.events import DuckEvent
from ..model.lane import STEM_LANES
from .context import Ctx

DUCK_KEYS = ("id", "target", "key", "at", "bars", "depth_db", "threshold_db", "attack_ms", "release_ms", "stem")
DEFAULTS = {"depth_db": 6.0, "threshold_db": -30.0, "attack_ms": 5.0, "release_ms": 120.0}


def duck_key(cx: Ctx, key, target: str, clips: dict[str, Clip], path: str) -> str | None:
    if not isinstance(key, str):
        cx.add("wrong_type", path, "expected a string", "\"self:vocals\" or a clip id")
        return None
    head, _, stem = key.partition(":")
    if head in ("self", target):
        if stem not in STEM_LANES:
            cx.add("unknown_stem", path, f"a self key needs a stem, got {key!r}", f"\"self:<stem>\", stem one of {list(STEM_LANES)}")
            return None
        if not clips[target].stems:
            cx.add("stems_off", path, "a stem key needs stems=true on the target clip", "set \"stems\": true (after separate_stems)")
            return None
        return f"self:{stem}"
    if stem or head not in clips:
        cx.add("unknown_clip", path, f"unknown key clip {key!r}" if not stem else "a key from another deck is the whole deck: use its clip id only",
               f"use \"self:<stem>\" or one of {list(clips)}")
        return None
    return head


def parse_ducks(cx: Ctx, items, clips: list[Clip]) -> list[DuckEvent]:
    out: list[DuckEvent] = []
    if not isinstance(items, list):
        cx.add("wrong_type", "plan.ducks", "expected a list")
        return out
    by_id = {c.id: c for c in clips}
    for j, d in enumerate(items):
        p = f"plan.ducks[{j}]"
        if not cx.obj(d, p, DUCK_KEYS, ("target", "key", "at", "bars")):
            continue
        if d["target"] not in by_id:
            cx.add("unknown_clip", f"{p}.target", f"unknown clip {d['target']!r}", f"one of {list(by_id)}")
            continue
        tgt = by_id[d["target"]]
        stem = d.get("stem")
        if stem is not None and (stem not in STEM_LANES or not tgt.stems):
            cx.add("unknown_stem", f"{p}.stem", "stem must be a stem name and the target must play stems",
                   f"one of {list(STEM_LANES)} with stems=true")
            continue
        key = duck_key(cx, d["key"], tgt.id, by_id, f"{p}.key")
        at, dur = cx.out_bars(d["at"], f"{p}.at", tgt.id), cx.length(d["bars"], f"{p}.bars", 1e-6, None)
        v = {k: cx.num(d.get(k, x), f"{p}.{k}", key=k) for k, x in DEFAULTS.items()}
        if key and key.startswith("self:") and stem is None:
            cx.add("stem_required", f"{p}.stem", "a self-keyed duck lowers one stem under another: name the target stem",
                   f"one of {list(STEM_LANES)}")
            continue
        if key and None not in (at, dur, *v.values()):
            out.append(DuckEvent(str(d.get("id") or f"dk{len(out) + 1}"), tgt.id, key, at * BAR, dur * BAR, **v, stem=stem))
    return out
