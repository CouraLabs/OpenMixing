from __future__ import annotations

from ..core.errors import OpenMixingError
from ..core.units import BAR
from ..dsp import sample, synth
from ..model.clip import Clip
from ..model.events import Trigger
from ..model.sampler import check_voices
from .context import Ctx

TRIGGER_KEYS = ("id", "kind", "at", "bars", "align", "gain_db", "params", "slot", "group")
ALIGNS = ("start", "end")


def check_params(cx: Ctx, kind: str, params, path: str) -> dict | None:
    if not isinstance(params, dict):
        cx.add("wrong_type", path, "expected an object")
        return None
    spec = sample.param_spec(kind)
    ok = True
    for k, v in params.items():
        if k not in spec:
            cx.add("unknown_param", f"{path}.{k}", f"{kind} has no parameter {k!r}", f"allowed: {sorted(spec)}")
            ok = False
        elif cx.num(v, f"{path}.{k}", spec[k][1], spec[k][2]) is None:
            ok = False
    return dict(params) if ok else None


def parse_triggers(cx: Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.triggers", "expected a list")
        return
    for j, t in enumerate(items):
        tp = f"{p}.triggers[{j}]"
        if not cx.obj(t, tp, TRIGGER_KEYS, ("kind", "at")):
            continue
        kind = t["kind"]
        if kind not in sample.KINDS:
            cx.add("unknown_kind", f"{tp}.kind", f"unknown generator {kind!r}", f"one of {sample.KINDS}")
            continue
        align = t.get("align", synth.KINDS.get(kind, {}).get("align", "start"))
        if align not in ALIGNS:
            cx.add("bad_align", f"{tp}.align", f"align must be one of {list(ALIGNS)}, got {align!r}", "use \"start\" or \"end\"")
            continue
        slot = t.get("slot")
        if (kind == "sample") != (slot is not None) or (slot is not None and slot not in cx.slots):
            cx.add("bad_slot", f"{tp}.slot", f"slot {slot!r} does not fit a {kind} trigger (sample triggers need a slot defined in plan.sampler)",
                   f"defined slots: {sorted(cx.slots)}")
            continue
        group = t.get("group")
        if group is not None and not isinstance(group, str):
            cx.add("wrong_type", f"{tp}.group", "expected a string")
            continue
        at, dur = cx.out_bars(t["at"], f"{tp}.at"), cx.length(t.get("bars", 1.0), f"{tp}.bars", 1e-6, None)
        gain, params = cx.num(t.get("gain_db", -6.0), f"{tp}.gain_db", key="gain_db"), check_params(cx, kind, t.get("params", {}), f"{tp}.params")
        if None in (at, dur, gain, params):
            continue
        if at > L + 1e-9 or (align == "end" and at < dur - 1e-9):
            cx.add("out_of_range", f"{tp}.at", f"the {kind} would start before the clip or at {at:g} beyond its {L:g} bars",
                   "keep the whole generator inside the clip start")
            continue
        clip.triggers.append(Trigger(str(t.get("id") or f"tg{len(clip.triggers) + 1}"), kind, at * BAR, dur * BAR, align, gain, params, slot, group))
    try:
        check_voices([(t.start, t.start + t.duration_beats) for t in clip.triggers if t.kind == "sample"], f"{p}.triggers")
    except OpenMixingError as e:
        cx.add(e.code, e.field, e.message, e.fix)
