from __future__ import annotations

from ..core.errors import OpenMixingError
from ..model.sampler import KINDS, SampleSlot, check_seconds
from .context import Ctx

SLOT_KEYS = ("path", "kind", "root_bpm", "gain_db", "pitch_semitones", "choke_group")


def parse_slot(cx: Ctx, d, p: str) -> SampleSlot | None:
    if not cx.obj(d, p, SLOT_KEYS, ("path",)):
        return None
    kind = d.get("kind", "oneshot")
    if kind not in KINDS:
        cx.add("bad_kind", f"{p}.kind", f"unknown sample kind {kind!r}", f"one of {list(KINDS)}")
        return None
    if not isinstance(d["path"], str):
        cx.add("wrong_type", f"{p}.path", "expected a file path string")
        return None
    root = None if d.get("root_bpm") is None else cx.num(d["root_bpm"], f"{p}.root_bpm", key="root_bpm")
    gain = cx.num(d.get("gain_db", 0.0), f"{p}.gain_db", key="gain_db")
    semi = cx.num(d.get("pitch_semitones", 0.0), f"{p}.pitch_semitones", key="sample_semitones")
    group = d.get("choke_group")
    if group is not None and not isinstance(group, str):
        cx.add("wrong_type", f"{p}.choke_group", "expected a string")
        return None
    try:
        path = cx.lib.sample_path(d["path"], f"{p}.path")
        check_seconds(kind, cx.lib.sample_seconds(path, f"{p}.path"), f"{p}.path")
    except OpenMixingError as e:
        cx.add(e.code, e.field or f"{p}.path", e.message, e.fix)
        return None
    if None in (gain, semi) or (d.get("root_bpm") is not None and root is None):
        return None
    return SampleSlot(str(path), kind, root, gain, semi, group)


def parse_sampler(cx: Ctx, items) -> dict[str, SampleSlot]:
    if not isinstance(items, dict):
        cx.add("wrong_type", "plan.sampler", "expected an object of slot name -> slot")
        return {}
    out = {}
    for name, d in items.items():
        if (slot := parse_slot(cx, d, f"plan.sampler.{name}")) is not None:
            out[name] = slot
    return out
