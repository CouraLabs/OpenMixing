from __future__ import annotations

from ..mixing.keymatch import MODES, plan_key_match
from .context import Ctx


def resolve_key_matches(cx: Ctx, lib, raw: list, clips: list) -> None:
    by_id = {c.id: c for c in clips}
    for i, c in enumerate(raw):
        clip = by_id.get(c.get("id", f"c{i + 1}")) if isinstance(c, dict) else None
        if clip is None or "key_match" not in c:
            continue
        p, km = f"clips[{i}].key_match", c["key_match"]
        if not cx.obj(km, p, ("to", "mode", "max_shift"), ("to",)):
            continue
        if "semitones" in c:
            cx.add("conflicting_fields", p, "key_match computes semitones", "remove semitones or key_match")
        ref = by_id.get(km["to"]) if isinstance(km["to"], str) else None
        mode = km.get("mode", "nearest_compatible")
        shift = cx.num(km.get("max_shift", 3), f"{p}.max_shift", 0, 12)
        if ref is None or ref is clip:
            cx.add("unknown_clip", f"{p}.to", f"no other clip {km['to']!r} in this plan", f"clip ids: {sorted(set(by_id) - {clip.id})}")
        elif mode not in MODES:
            cx.add("invalid_value", f"{p}.mode", f"unknown mode {mode!r}", f"one of {list(MODES)}")
        elif shift is not None:
            r = plan_key_match(lib.effective(ref.track), lib.effective(clip.track), int(shift), mode)
            if not r["low_confidence"]:
                clip.semitones = r["semitones_in"]
