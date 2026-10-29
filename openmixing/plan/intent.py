from __future__ import annotations

from ..validate import INTENT_KEYS, SHAPES
from .context import Ctx

TENSION_KEYS = ("shape", "points", "min_correlation", "start_sec", "end_sec")


def parse_tension(cx: Ctx, v) -> dict:
    p = "plan.intent.tension_target"
    if not cx.obj(v, p, TENSION_KEYS):
        return {}
    if ("shape" in v) == ("points" in v):
        cx.add("invalid_value", p, "give exactly one of shape or points", f"shape is one of {list(SHAPES)}; points is [[sec, level], ...]")
        return {}
    out = {}
    if "shape" in v:
        if v["shape"] in SHAPES:
            out["shape"] = v["shape"]
        else:
            cx.add("invalid_value", f"{p}.shape", f"unknown shape {v['shape']!r}", f"use one of {list(SHAPES)}")
    else:
        pts = v["points"]
        if isinstance(pts, list) and len(pts) >= 2 and all(isinstance(q, list) and len(q) == 2 for q in pts):
            nums = [[cx.num(q[0], f"{p}.points[{i}][0]", 0.0, None), cx.num(q[1], f"{p}.points[{i}][1]", key="tension_level")]
                    for i, q in enumerate(pts)]
            if all(a is not None and b is not None for a, b in nums):
                out["points"] = nums
        else:
            cx.add("wrong_type", f"{p}.points", "expected at least two [sec, level] pairs", "e.g. [[0, 0.1], [30, 0.9]]")
    for k, key, lo in (("min_correlation", "correlation", None), ("start_sec", None, 0.0), ("end_sec", None, 0.0)):
        if k in v and (n := cx.num(v[k], f"{p}.{k}", lo, None, key)) is not None:
            out[k] = n
    return out


def parse_intent(cx: Ctx, v) -> dict:
    if v is None:
        return {}
    if not cx.obj(v, "plan.intent", INTENT_KEYS):
        return {}
    out = {}
    for k in ("beat_drift_ms_max", "low_end_overlap_max_sec", "vocal_overlap_max_sec", "pre_drop_gap_min_beats"):
        if k in v and (n := cx.num(v[k], f"plan.intent.{k}", 0.0, None)) is not None:
            out[k] = n
    if "true_peak_max_dbtp" in v and (n := cx.num(v["true_peak_max_dbtp"], "plan.intent.true_peak_max_dbtp", -60.0, 6.0)) is not None:
        out["true_peak_max_dbtp"] = n
    if "harmonic_dissonance_max" in v and (n := cx.num(v["harmonic_dissonance_max"], "plan.intent.harmonic_dissonance_max",
                                                       key="dissonance")) is not None:
        out["harmonic_dissonance_max"] = n
    if "tension_target" in v and (t := parse_tension(cx, v["tension_target"])):
        out["tension_target"] = t
    for k in ("beat_drift_intended", "low_end_clash_intended", "vocal_clash_intended", "harmonic_clash_intended"):
        if k in v:
            if isinstance(v[k], bool):
                out[k] = v[k]
            else:
                cx.add("wrong_type", f"plan.intent.{k}", "expected true or false")
    if "loudness_jump_lu" in v:
        j = v["loudness_jump_lu"]
        if isinstance(j, list) and len(j) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in j) and j[0] <= j[1]:
            out["loudness_jump_lu"] = [float(j[0]), float(j[1])]
        else:
            cx.add("wrong_type", "plan.intent.loudness_jump_lu", "expected [min, max] with min <= max", "e.g. [-1.5, 1.5]")
    if "silence_intended" in v:
        sv = v["silence_intended"]
        span = lambda r: isinstance(r, list) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r) and r[0] < r[1]
        ok = isinstance(sv, list) and all(span(r) for r in sv)
        if ok:
            out["silence_intended"] = [[float(a), float(b)] for a, b in sv]
        else:
            cx.add("wrong_type", "plan.intent.silence_intended", "expected a list of [start_sec, end_sec]", "e.g. [[30.0, 31.0]]")
    return out
