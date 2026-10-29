from __future__ import annotations

from ...core.errors import OpenMixingError
from ...core.units import BAR
from ...model.clip import Clip
from ...model.events import FxEvent, LoopEvent, Trigger
from ...model.lane import MIN_RAMP, Lane
from ...model.roll import HIT_BEATS, expand_roll
from .gap import MIN_GAP, hold_gap
from .tension import LOOP_ROLL_MIN_LEVEL, level_fn, mapped

ELEMENTS = ("riser", "filter_sweep", "hp_eq", "volume_swell", "snare_roll", "loop_roll")
LOOP_ROLL = ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.5))
ROLL_BARS = 2.0
LANE_ELEMENTS = (("filter_sweep", "filter"), ("hp_eq", "eq_low"))
CATALOG = {"build_and_release": "Ramp tension into a resolution downbeat: rising high-pass, bass kill, volume swell, riser, snare roll, loop roll, "
                                "a short silent gap, a reverb throw and an impact on the downbeat; every lane returns to neutral there."}


def _sweep(lane: Lane, a: float, length: float, fn) -> None:
    before = lane.value_at(a)
    lane.curve(a, length, fn)
    if a > MIN_RAMP:
        lane.points.append([a - MIN_RAMP, before])
    lane.points.sort(key=lambda p: p[0])


def _fail(code: str, field: str, message: str, fix: str) -> OpenMixingError:
    return OpenMixingError(code, message, field, fix)


def build_and_release(clip: Clip, resolve_bar: float, tid: str, build_bars: float = 8.0, elements=None, peak: float = 0.9, gap_beats: float = 0.5,
                      impact: bool = True, reverb_throw: bool = True, tension: list[list[float]] | None = None) -> dict:
    names = tuple(ELEMENTS if elements is None else elements)
    bad = [e for e in names if e not in ELEMENTS]
    if bad:
        raise _fail("unknown_technique", "elements", f"unknown element {bad[0]!r}", f"one of {list(ELEMENTS)}")
    R, nb, gap, group = resolve_bar * BAR, build_bars * BAR, float(gap_beats), f"tension:{tid}"
    if R - nb < -1e-9 or R > clip.length_beats + 1e-9:
        raise _fail("out_of_range", "resolve_at", f"a {build_bars:g}-bar build ending at bar {resolve_bar:g} does not fit the clip "
                    f"(0..{clip.length_beats / BAR:g} bars)", "shorten build_bars or resolve later")
    if gap and not MIN_GAP <= gap <= nb - 1.0 - 1e-9:
        raise _fail("out_of_range", "gap_beats", f"gap_beats {gap:g} must be 0 or between {MIN_GAP:g} and {nb - 1:g}", "use 0 for no gap")
    B0, end = R - nb, R - gap
    lv = level_fn(tension, B0, nb, peak)
    for el, lane in LANE_ELEMENTS:
        if el in names:
            _sweep(clip.lanes[lane], B0, nb, lambda p, lane=lane: mapped(lane, lv(p)))
            clip.lanes[lane].set(R, 0.0, MIN_RAMP)
    if "volume_swell" in names:
        _sweep(clip.lanes["volume"], B0, nb - gap, lambda p: mapped("volume", lv(p)))
        if not gap:
            clip.lanes["volume"].set(R, 1.0, MIN_RAMP)
    hold_gap(clip.lanes["volume"], R, gap)
    if "riser" in names:
        clip.triggers.append(Trigger(f"{tid}_riser", "riser", end, end - B0, "end", mapped("riser_db", lv(1.0)), {}, None, group))
    if "snare_roll" in names:
        stop = end - HIT_BEATS if gap else R
        length = min(ROLL_BARS * BAR, stop - B0)
        hits = expand_roll(stop - length, length, 0.5, 0.0625, "log", mapped("roll_db", lv(1 - length / nb)), mapped("roll_db", lv(1.0)), "snare", {})
        for k, h in enumerate(hits):
            h.id, h.group = f"{tid}_sn{k}", group
        clip.triggers += hits
    if "loop_roll" in names and lv(1.0) >= LOOP_ROLL_MIN_LEVEL:
        t = R - BAR
        for k, (size, ln) in enumerate(LOOP_ROLL):
            clip.loops.append(LoopEvent(f"{tid}_lp{k}", t, size, ln, True, group))
            t += ln
        try:
            clip.validate_loops()
            clip.validate_rate()
        except ValueError as e:
            raise _fail("loop_overlap", "elements", str(e), "drop loop_roll or move the resolution") from e
    if reverb_throw:
        clip.fx.append(FxEvent(f"{tid}_reverb", "reverb", end - 1.0, 1.0, 0.25, 0.4, {}, None, None, None, group))
    if impact:
        clip.triggers.append(Trigger(f"{tid}_impact", "impact", R, BAR, "start", -4.0, {}, None, group))
    return {"group": group, "build_starts_at_bar": B0 / BAR, "resolves_at_bar": R / BAR, "gap_beats": gap, "elements": list(names),
            "triggers": sum(t.group == group for t in clip.triggers), "loops": sum(l.group == group for l in clip.loops)}


def remove_group(clip: Clip, group: str) -> dict:
    ev = [*(e for e in clip.fx if e.group == group), *(e for e in clip.loops if e.group == group), *(e for e in clip.triggers if e.group == group)]
    if not ev:
        raise _fail("not_found", "group", f"no events tagged {group!r} on clip {clip.id}", "see the group field in get_set / export_plan")
    a = min(getattr(e, "start", e.beat) for e in ev)
    b = max(getattr(e, "start", e.beat) + (getattr(e, "length_beats", None) or e.duration_beats) for e in ev)
    clip.fx, clip.loops, clip.triggers = ([e for e in xs if e.group != group] for xs in (clip.fx, clip.loops, clip.triggers))
    for lane in ("filter", "eq_low", "volume"):
        clip.lanes[lane].clear(a, b)
    return {"clip": clip.id, "removed": len(ev), "lanes_reset_between_bars": [a / BAR, b / BAR]}
