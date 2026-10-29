from __future__ import annotations

import math

from ..core.units import BAR, BEAT_DIVISIONS, BEAT_LABELS, parse_beats
from ..dsp import fx as fxm
from ..model.clip import Clip
from ..model.events import FxEvent
from ..model.lane import STEM_LANES
from .context import Ctx
from .ranges import MAX_CHAIN

STEP_KEYS = ("fx", "beats", "wet", "params")


def parse_step(cx: Ctx, f: dict, fp: str, spec) -> tuple | None:
    try:
        beats = parse_beats(f.get("beats", 1.0))
    except (ValueError, TypeError):
        beats = None
        cx.add("bad_beats", f"{fp}.beats", f"beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {f.get('beats')!r}",
               "use a tempo-sync division")
    wet = cx.num(f.get("wet", 1.0), f"{fp}.wet", key="wet")
    params = f.get("params", {})
    if not isinstance(params, dict):
        cx.add("wrong_type", f"{fp}.params", "expected an object")
        params = {}
    for k, v in params.items():
        if spec is not None and k not in spec.params:
            cx.add("unknown_param", f"{fp}.params.{k}", f"{spec.name} has no parameter {k!r}", f"allowed: {sorted(spec.params)}")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            cx.add("wrong_type", f"{fp}.params.{k}", "effect parameters must be finite numbers")
    return (beats, wet, dict(params)) if spec and None not in (beats, wet) else None


def spec_of(cx: Ctx, name, path: str):
    spec = fxm.CATALOG.get(name) if isinstance(name, str) else None
    if spec is None:
        cx.add("unknown_fx", path, f"unknown effect {name!r}", f"one of {list(fxm.CATALOG)}")
    return spec


def parse_chain(cx: Ctx, items, fp: str) -> list[dict] | None:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_CHAIN:
        cx.add("out_of_range", f"{fp}.chain", f"expected a list of 1 to {MAX_CHAIN} effects", "give 1 to 6 steps")
        return None
    steps = []
    for k, s in enumerate(items):
        sp = f"{fp}.chain[{k}]"
        if not cx.obj(s, sp, STEP_KEYS, ("fx",)):
            continue
        spec = spec_of(cx, s["fx"], f"{sp}.fx")
        if r := parse_step(cx, s, sp, spec):
            steps.append({"fx": spec.name, "beats": r[0], "wet": r[1], "params": r[2]})
    return steps if len(steps) == len(items) else None


def parse_wet_points(cx: Ctx, items, fp: str) -> list[list[float]] | None:
    if not isinstance(items, list) or not items or not all(isinstance(p, list) and len(p) == 2 for p in items):
        cx.add("wrong_type", f"{fp}.wet_points", "expected a list of [bar, wet] pairs", "e.g. [[0, 0], [4, 1]]")
        return None
    pts = [[cx.length(p[0], f"{fp}.wet_points[{j}][0]", 0.0, None), cx.num(p[1], f"{fp}.wet_points[{j}][1]", key="wet")] for j, p in enumerate(items)]
    if any(None in p for p in pts):
        return None
    if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        cx.add("not_ascending", f"{fp}.wet_points", "bars must increase", "sort the points by bar")
        return None
    return [[p[0] * BAR, p[1]] for p in pts]


def parse_fx(cx: Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.fx", "expected a list")
        return
    for j, f in enumerate(items):
        fp = f"{p}.fx[{j}]"
        if not cx.obj(f, fp, ("id", "at", "bars", "stem", "wet_points", "chain", "group", *STEP_KEYS), ("at", "bars")):
            continue
        if ("fx" in f) == ("chain" in f):
            cx.add("fx_or_chain", fp, "give exactly one of fx / chain", "remove one of them")
            continue
        chain = parse_chain(cx, f["chain"], fp) if "chain" in f else None
        spec = spec_of(cx, f["fx"], f"{fp}.fx") if "fx" in f else None
        stem = f.get("stem")
        if stem is not None:
            if stem not in STEM_LANES:
                cx.add("unknown_stem", f"{fp}.stem", f"unknown stem {stem!r}", f"one of {list(STEM_LANES)}")
                continue
            if not clip.stems:
                cx.add("stems_off", f"{fp}.stem", "a stem effect needs stems=true on this clip",
                       "set \"stems\": true (after separate_stems) or drop \"stem\"")
                continue
        at, dur = cx.out_bars(f["at"], f"{fp}.at"), cx.length(f["bars"], f"{fp}.bars", 1e-6, None)
        if at is not None and at > L + 1e-9:
            cx.add("out_of_range", f"{fp}.at", f"{at:g} is beyond the clip ({L:g} bars)", "move it inside the clip")
        group = f.get("group")
        if group is not None and not isinstance(group, str):
            cx.add("wrong_type", f"{fp}.group", "expected a string")
            continue
        curve = parse_wet_points(cx, f["wet_points"], fp) if "wet_points" in f else None
        step = parse_step(cx, f, fp, spec) if chain is None else (1.0, cx.num(f.get("wet", 1.0), f"{fp}.wet", key="wet"), {})
        if step and None not in (at, dur, step[1]) and (chain or spec) and not ("wet_points" in f and curve is None):
            name = "chain" if chain else spec.name
            clip.fx.append(FxEvent(str(f.get("id") or f"fx{len(clip.fx) + 1}"), name, at * BAR, dur * BAR, step[0], step[1], step[2], stem,
                                   chain, curve, group))
