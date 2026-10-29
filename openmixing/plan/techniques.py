from __future__ import annotations

from ..core.errors import OpenMixingError
from ..core.units import BAR
from ..mixing.techniques import CATALOG, ELEMENTS, build_and_release, existing_gap
from ..model.clip import Clip
from .context import Ctx

TECHNIQUE_KEYS = ("technique", "id", "resolve_at", "build_bars", "elements", "peak", "gap_beats", "impact", "reverb_throw", "respect_existing_gap",
                  "tension")


def parse_tension(cx: Ctx, v, path: str) -> list[list[float]] | None:
    if not isinstance(v, list) or len(v) < 2 or not all(isinstance(p, list) and len(p) == 2 for p in v):
        cx.add("wrong_type", path, "expected at least two [bar, level] pairs", "e.g. [[0, 0], [8, 1]]")
        return None
    pts = [[cx.out_bars(p[0], f"{path}[{k}][0]"), cx.num(p[1], f"{path}[{k}][1]", key="tension_level")] for k, p in enumerate(v)]
    if any(None in p for p in pts):
        return None
    if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        cx.add("not_ascending", path, "bars must increase", "sort the points by bar")
        return None
    return [[b * BAR, lv] for b, lv in pts]


def parse_techniques(cx: Ctx, items, p: str, clip: Clip, L: float, sections: list[dict]) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.techniques", "expected a list")
        return
    for j, t in enumerate(items):
        tp = f"{p}.techniques[{j}]"
        if not cx.obj(t, tp, TECHNIQUE_KEYS, ("technique", "resolve_at")):
            continue
        if t["technique"] not in CATALOG:
            cx.add("unknown_technique", f"{tp}.technique", f"unknown technique {t['technique']!r}", f"one of {list(CATALOG)}")
            continue
        at = cx.out_bars(t["resolve_at"], f"{tp}.resolve_at")
        build = cx.length(t.get("build_bars", 8.0), f"{tp}.build_bars", 1.0, None)
        peak = cx.num(t.get("peak", 0.9), f"{tp}.peak", key="tension_level")
        gap = cx.num(t.get("gap_beats", 0.5), f"{tp}.gap_beats", key="gap_beats")
        flags = {k: t.get(k, True) for k in ("impact", "reverb_throw", "respect_existing_gap")}
        if not all(isinstance(v, bool) for v in flags.values()):
            cx.add("wrong_type", tp, "impact, reverb_throw and respect_existing_gap must be true or false")
            continue
        tension = parse_tension(cx, t["tension"], f"{tp}.tension") if "tension" in t else None
        elements = t.get("elements")
        if None in (at, build, peak, gap) or ("tension" in t and tension is None):
            continue
        if flags["respect_existing_gap"] and existing_gap(sections, clip.src_start_beat / BAR + at):
            gap = 0.0
        try:
            build_and_release(clip, at, str(t.get("id") or f"t{j + 1}"), build, elements, peak, gap, flags["impact"], flags["reverb_throw"], tension)
        except OpenMixingError as e:
            cx.add(e.code, f"{tp}.{e.field}" if e.field else tp, e.message, e.fix)
        except TypeError:
            cx.add("wrong_type", f"{tp}.elements", "expected a list of element names", f"one of {list(ELEMENTS)}")
