from __future__ import annotations

from ..core.units import BAR
from ..model.clip import Clip
from ..model.events import RateEvent
from ..model.rate import KINDS, check_curve, preset
from .context import Ctx
from .ranges import RANGES


def parse_rate(cx: Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.rate", "expected a list")
        return
    lo, hi = RANGES["rate_beats"]
    for j, r in enumerate(items):
        rp = f"{p}.rate[{j}]"
        if not cx.obj(r, rp, ("id", "kind", "at", "bars", "points", "slip"), ("at", "bars")):
            continue
        kind = r.get("kind", "custom")
        if kind not in KINDS:
            cx.add("unknown_kind", f"{rp}.kind", f"unknown rate kind {kind!r}", f"one of {list(KINDS)}")
            continue
        at, dur = cx.out_bars(r["at"], f"{rp}.at"), cx.length(r["bars"], f"{rp}.bars", lo / BAR, hi / BAR)
        slip = r.get("slip", False)
        if not isinstance(slip, bool):
            cx.add("wrong_type", f"{rp}.slip", "expected true or false")
            continue
        try:
            pts = [[float(u), float(v)] for u, v in r["points"]] if "points" in r else preset(kind)
            check_curve(pts, *RANGES["rate_speed"])
        except (ValueError, TypeError) as e:
            cx.add("bad_curve", f"{rp}.points", str(e), "give [[0, speed], ..., [1, speed]] with speeds in the allowed range")
            continue
        if at is not None and at > L + 1e-9:
            cx.add("out_of_range", f"{rp}.at", f"{at:g} is beyond the clip ({L:g} bars)", "move it inside the clip")
        elif None not in (at, dur):
            clip.rate.append(RateEvent(str(r.get("id") or f"rt{len(clip.rate) + 1}"), at * BAR, dur * BAR, pts, slip, kind))
    try:
        clip.validate_rate()
    except ValueError as e:
        cx.add("rate_overlap", f"{p}.rate", str(e), "rate events may not overlap each other or a loop")
