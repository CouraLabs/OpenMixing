from __future__ import annotations

from ..core.units import BAR
from ..dsp.eq import KILL_DB
from ..model.curves import CURVES, write
from ..model.lane import Lane
from .context import Ctx


def parse_envelope(cx: Ctx, v, path: str, lane: Lane, value_key: str, lo: float, hi: float, length: float, kill: bool = False) -> None:
    if isinstance(v, dict):
        if not cx.obj(v, path, ("points",), ("points",)):
            return
        pts = v["points"]
        if not isinstance(pts, list):
            cx.add("wrong_type", f"{path}.points", "expected a list of [at, value] pairs")
            return
        out = []
        for j, p in enumerate(pts):
            pp = f"{path}.points[{j}]"
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                cx.add("wrong_type", pp, "expected [at, value]")
                continue
            a, val = cx.out_bars(p[0], f"{pp}[0]"), cx.num(p[1], f"{pp}[1]", lo, hi)
            if a is not None and a > length + 1e-9:
                cx.add("out_of_range", f"{pp}[0]", f"{a:g} is beyond the clip ({length:g} bars)", "shorten the envelope or lengthen the clip")
            elif a is not None and val is not None:
                out.append([a * BAR, val])
        if sorted(p[0] for p in out) != [p[0] for p in out]:
            cx.add("unsorted_points", f"{path}.points", "points must be in increasing time order", "sort by the first element")
        lane.points = out
        return
    if not isinstance(v, list):
        cx.add("wrong_type", path, "expected a list of events or {\"points\": [...]}", "see get_plan_schema")
        return
    evs = []
    for j, e in enumerate(v):
        ep = f"{path}[{j}]"
        if not cx.obj(e, ep, ("at", value_key, "ramp_bars", "curve", "exponent"), ("at", value_key)):
            continue
        at = cx.out_bars(e["at"], f"{ep}.at")
        if at is not None and at > length + 1e-9:
            cx.add("out_of_range", f"{ep}.at", f"{at:g} is beyond the clip ({length:g} bars)", "move it inside the clip")
            at = None
        raw = e[value_key]
        if kill and raw == "kill":
            val = KILL_DB
        else:
            val = cx.num(raw, f"{ep}.{value_key}", lo, hi)
        ramp = cx.length(e.get("ramp_bars", 0.0), f"{ep}.ramp_bars", key="ramp_bars")
        curve = e.get("curve", "linear")
        if curve not in CURVES:
            cx.add("unknown_curve", f"{ep}.curve", f"unknown curve {curve!r}", f"one of {list(CURVES)}")
            curve = None
        k = cx.num(e.get("exponent", 2.0), f"{ep}.exponent", key="exponent")
        if "exponent" in e and curve not in (None, "power"):
            cx.add("unused_field", f"{ep}.exponent", "exponent only applies to curve 'power'", "remove it or use curve 'power'")
        if None not in (at, val, ramp, curve, k):
            evs.append((at, val, ramp, curve, k))
    for at, val, ramp, curve, k in sorted(evs, key=lambda e: e[0]):
        write(lane, at * BAR, val, ramp * BAR, curve, k)
