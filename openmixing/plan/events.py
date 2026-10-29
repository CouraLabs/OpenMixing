from __future__ import annotations

from ..core.units import BAR, BEAT_DIVISIONS, BEAT_LABELS, parse_beats
from ..model.clip import Clip
from ..model.events import LoopEvent
from .context import Ctx


def parse_loops(cx: Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.loops", "expected a list")
        return
    for j, l in enumerate(items):
        lp = f"{p}.loops[{j}]"
        if not cx.obj(l, lp, ("id", "at", "size_beats", "bars", "repeats", "roll", "group"), ("at", "size_beats")):
            continue
        at = cx.out_bars(l["at"], f"{lp}.at")
        try:
            size = parse_beats(l["size_beats"])
        except (ValueError, TypeError):
            size = None
            cx.add("bad_beats", f"{lp}.size_beats", f"size_beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {l['size_beats']!r}",
                   "use a loop size from the list")
        if ("bars" in l) == ("repeats" in l):
            cx.add("bars_or_repeats", lp, "give exactly one of bars / repeats", "remove one of them")
            continue
        if "bars" in l:
            n = cx.length(l["bars"], f"{lp}.bars", 1e-6, None)
            length = None if n is None else n * BAR
        else:
            r = l["repeats"]
            if isinstance(r, bool) or not isinstance(r, int) or r < 1:
                cx.add("out_of_range", f"{lp}.repeats", f"repeats must be an integer >= 1, got {r!r}", "use a whole number")
                length = None
            else:
                length = None if size is None else size * r
        roll = l.get("roll", False)
        if not isinstance(roll, bool):
            cx.add("wrong_type", f"{lp}.roll", "expected true or false")
            roll = False
        group = l.get("group")
        if group is not None and not isinstance(group, str):
            cx.add("wrong_type", f"{lp}.group", "expected a string")
            group = None
        if at is not None and at > L + 1e-9:
            cx.add("out_of_range", f"{lp}.at", f"{at:g} is beyond the clip ({L:g} bars)", "move it inside the clip")
        elif None not in (at, size, length):
            clip.loops.append(LoopEvent(str(l.get("id") or f"lp{len(clip.loops) + 1}"), at * BAR, size, length, roll, group))
    try:
        clip.validate_loops()
    except ValueError as e:
        cx.add("loop_overlap", f"{p}.loops", str(e), "loops may not overlap: shorten one or move it")
