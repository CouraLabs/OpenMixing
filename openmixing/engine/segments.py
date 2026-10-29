from __future__ import annotations

from ..model.clip import Clip
from ..model.events import RateEvent, Trigger
from ..model.rate import advance
from .triggers import LOOKBACK_BEATS


def segments(clip: Clip) -> list[tuple]:
    clip.validate_loops()
    clip.validate_rate()
    segs, cur_o, cur_s = [], 0.0, clip.src_start_beat
    for L in sorted([*clip.loops, *clip.rate], key=lambda l: l.beat):
        if L.beat >= clip.length_beats:
            break
        if L.beat > cur_o:
            segs.append(("play", cur_o, L.beat, cur_s))
            cur_s += L.beat - cur_o
            cur_o = L.beat
        if isinstance(L, RateEvent):
            end = min(L.beat + L.duration_beats, clip.length_beats)
            segs.append(("rate", cur_o, end, cur_s, L))
            cur_s += end - cur_o if L.slip else advance(L, end - cur_o)
            cur_o = end
            continue
        end = min(L.beat + L.length_beats, clip.length_beats)
        segs.append(("loop", cur_o, end, cur_s, L.size_beats))
        run = end - cur_o
        r = run % L.size_beats
        if r < 1e-9 or L.size_beats - r < 1e-9:
            r = L.size_beats
        cur_s += run if L.roll else min(r, run)
        cur_o = end
    if cur_o < clip.length_beats:
        segs.append(("play", cur_o, clip.length_beats, cur_s))
    return segs


def expand_window(clip: Clip, wa: float, wb: float) -> tuple[float, float]:
    wa = max(wa - 2.0, 0.0)
    changed = True
    while changed:
        changed = False
        for ev in [*clip.fx, *clip.loops, *clip.rate, *clip.triggers]:
            dur = getattr(ev, "duration_beats", None) or ev.length_beats
            begin = ev.start if isinstance(ev, Trigger) else ev.beat
            start = max(begin - LOOKBACK_BEATS, 0.0) if isinstance(ev, Trigger) else begin
            if start < wa < begin + dur + 1e-9:
                wa, changed = start, True
    return wa, min(wb, clip.length_beats)
