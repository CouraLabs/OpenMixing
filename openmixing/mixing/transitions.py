from __future__ import annotations

import numpy as np

from ..core.units import BAR
from ..model.clip import Clip
from ..model.events import FxEvent, LoopEvent, new_id
from ..model.lane import STEM_LANES
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap
from .compat import tempo_match
from .techniques import MIN_GAP, hold_gap

TEMPLATES = {
    "bass_swap": "Classic blend: incoming enters with bass killed, bass swaps at the midpoint, outgoing fades out. Best for same-genre, 16-32 bars.",
    "long_blend": "Slow equal-power blend with the incoming mids/highs eased in and bass swapped at 3/4. 32+ bars, harmonic matches only.",
    "filter_fade": "Outgoing is low-passed shut while the incoming opens from a high-passed state. Good across energy changes.",
    "echo_out": "Outgoing is thrown into an echo (dry cut) on the last bars and the incoming drops in full on the phrase boundary.",
    "drop_swap": "Hard swap on a phrase boundary: outgoing stops, incoming hits at full. For breakdown -> drop or genre jumps.",
    "loop_roll_out": "Outgoing rolls into shrinking loops (1, 1/2, 1/4, 1/8 beat) in its last bar, then the incoming drops.",
    "stem_blend": ("Stem-based blend (both clips need separated stems): outgoing vocals leave first, the harmonic layers cross over, then the "
                   "drums+bass swap, and the incoming vocals come in last. No bass or vocal ever doubles up; works across key clashes better "
                   "than an EQ blend."),
}

GAP_STYLES = ("echo_out", "drop_swap")
LANE_NAMES = ("eq_low", "eq_mid", "eq_high", "filter", "volume", *STEM_LANES.values())


def _reset_from(clip: Clip, beat: float) -> None:
    for l in LANE_NAMES:
        clip.lanes[l].points = [p for p in clip.lanes[l].points if p[0] < beat - 1e-9]


def clamp_to_clip(*clips: Clip) -> None:
    for c in clips:
        for lane in list(c.lanes.values()) + [c.tempo]:
            if lane.points and lane.points[-1][0] > c.length_beats:
                end = float(lane.value_at(c.length_beats))
                lane.points = [p for p in lane.points if p[0] < c.length_beats] + [[c.length_beats, end]]


def apply_template(model: SetModel, info, out: Clip, inc: Clip, style: str, at_bar: float | None, bars: float,
                   tempo_after: str = "keep", trim_out: bool = True, gap_beats: float = 0.0) -> dict:
    if style not in TEMPLATES:
        raise ValueError(f"style must be one of {list(TEMPLATES)}")
    if gap_beats and (style not in GAP_STYLES or not MIN_GAP <= gap_beats <= bars * BAR / 2):
        raise ValueError(f"gap_beats applies to {list(GAP_STYLES)} and must be 0 or between {MIN_GAP:g} and half the blend")
    nb = bars * BAR
    out_bars = out.length_beats / BAR
    at = float(out_bars - bars if at_bar is None else at_bar)
    if at < 0 or at * BAR >= out.length_beats:
        raise ValueError(f"at_bar {at} is outside clip {out.id} (0..{out_bars:g} bars)")
    a0 = at * BAR
    out_end = a0 + nb
    K = -80.0

    inc.start = {"anchor": out.id, "anchor_beat": out_end if style == "loop_roll_out" else a0}
    tm = TempoMap(out, float(info(out.track)["bpm"]))
    sync_bpm = float(tm.bpm_at(a0))
    native = float(info(inc.track)["bpm"])
    ratio = tempo_match(sync_bpm, native)["ratio_used"]
    play_bpm = sync_bpm / ratio
    inc.tempo.points = [[0.0, play_bpm]]
    if tempo_after == "native":
        inc.tempo.points += [[nb, play_bpm], [nb + 8 * BAR, native]]

    if style == "stem_blend":
        for c in (out, inc):
            if not info(c.track).get("stems"):
                raise ValueError(f"stem_blend needs stems for {info(c.track)['file']}: call separate_stems "
                                 + ("(force=True: it only has the old 3-stem layout)" if info(c.track).get("stems_legacy") else "first"))
            c.stems = True

    _reset_from(out, a0)
    for l in LANE_NAMES:
        inc.lanes[l].points = []
    notes = [f"{inc.id} enters {out.id} at bar {at:g} playing at {play_bpm:.2f} bpm (native {native:.2f})"]

    if style in ("bass_swap", "long_blend"):
        swap = nb * (0.5 if style == "bass_swap" else 0.75)
        inc.lanes["eq_low"].hold(K)
        inc.lanes["eq_low"].set(swap, 0.0, 0.5)
        out.lanes["eq_low"].set(a0 + swap, K, 0.5)
        if style == "long_blend":
            inc.lanes["eq_high"].hold(-12.0)
            inc.lanes["eq_high"].set(nb * 0.5, 0.0, nb * 0.25)
            inc.lanes["volume"].hold(0.0)
            inc.lanes["volume"].curve(0, nb, lambda p: np.sin(p * np.pi / 2))
            out.lanes["volume"].curve(a0, nb, lambda p: np.cos(p * np.pi / 2))
        else:
            inc.lanes["volume"].hold(0.0)
            inc.lanes["volume"].set(0, 1.0, nb * 0.4)
            out.lanes["volume"].set(a0 + nb * 0.6, 0.0, nb * 0.4)
        notes.append(f"incoming bass killed until beat {swap:g} of the blend, then the bass swaps; faders cross")
    elif style == "filter_fade":
        inc.lanes["filter"].hold(1.0)
        inc.lanes["filter"].set(nb * 0.25, 0.0, nb * 0.5)
        out.lanes["filter"].set(a0 + nb * 0.25, -1.0, nb * 0.6)
        inc.lanes["eq_low"].hold(K)
        inc.lanes["eq_low"].set(nb * 0.5, 0.0, 0.5)
        out.lanes["eq_low"].set(a0 + nb * 0.5, K, 0.5)
        out.lanes["volume"].set(a0 + nb * 0.85, 0.0, nb * 0.15)
        notes.append("outgoing low-passes shut, incoming high-pass opens; bass swap at the midpoint")
    elif style == "echo_out":
        fx_at = out_end - min(2 * BAR, nb)
        out.fx = [e for e in out.fx if not (a0 <= e.beat <= out_end)]
        out.fx.append(FxEvent(new_id("fx"), "echo_out", fx_at, out_end - fx_at, 1.0, 1.0, {}))
        inc.lanes["eq_low"].hold(K)
        inc.lanes["volume"].hold(0.0)
        if gap_beats:
            hold_gap(out.lanes["volume"], out_end, gap_beats, release=False)
            inc.lanes["eq_low"].set(nb, 0.0, 0.25)
            inc.lanes["volume"].set(nb, 1.0, 0.05)
            notes.append(f"echo_out (1 beat) on {out.id} from bar {fx_at / BAR:g}; {gap_beats:g} beats of silence (the echo tail rings) "
                         f"before {inc.id} hits at bar {out_end / BAR:g}")
        else:
            out.lanes["volume"].set(out_end, 0.0, 0.5)
            inc.lanes["eq_low"].set(fx_at - a0, 0.0, 0.25)
            inc.lanes["volume"].set(fx_at - a0 - 0.5, 1.0, 0.5)
            notes.append(f"echo_out (1 beat) on {out.id} from bar {fx_at / BAR:g}; incoming fades in just before, bass opens with the echo")
    elif style == "stem_blend":
        ramp = nb * 0.125
        def fade(c, stem, frac, to, base=0.0):
            c.lanes[STEM_LANES[stem]].set(base + nb * frac, to, ramp)
        for stem in STEM_LANES:
            inc.lanes[STEM_LANES[stem]].hold(0.0)
        fade(out, "vocals", 0.0, 0.0, a0)
        fade(inc, "other", 0.25, 1.0)
        fade(out, "other", 0.25, 0.0, a0)
        low_in_drums = any(info(c.track).get("bass_carrier") == "drums_lowband" for c in (out, inc))
        d_at = 0.5 if low_in_drums else 0.4
        fade(inc, "drums", d_at, 1.0)
        fade(out, "drums", d_at, 0.0, a0)
        inc.lanes[STEM_LANES["bass"]].set(nb * 0.5, 1.0, 0.25)
        out.lanes[STEM_LANES["bass"]].set(a0 + nb * 0.5, 0.0, 0.25)
        fade(inc, "vocals", 0.75, 1.0)
        notes.append(f"vocals out -> other crossover (25%) -> drums swap ({d_at:.0%}) -> bass swap (50%) -> incoming vocals (75%); "
                     "both clips now play stems")
        if low_in_drums:
            notes.append("the bass stem is empty on at least one track (808 sits in drums): the drums swap moves to the bass swap point")
    elif style == "drop_swap":
        if gap_beats:
            hold_gap(out.lanes["volume"], out_end, gap_beats, release=False)
        else:
            out.lanes["volume"].set(out_end, 0.0, 0.05)
        inc.start = {"anchor": out.id, "anchor_beat": out_end}
        notes.append(f"{out.id} stops at bar {(out_end - gap_beats) / BAR:g}"
                     + (f", {gap_beats:g} beats of silence," if gap_beats else "") + f" and {inc.id} hits at full level on bar {out_end / BAR:g}")
    elif style == "loop_roll_out":
        out.loops = [l for l in out.loops if l.beat + l.length_beats < a0 or l.beat >= out_end]
        t = out_end - BAR
        for size, ln in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.5)):
            out.loops.append(LoopEvent(new_id("lp"), t, size, ln, True))
            t += ln
        notes.append(f"loop roll 1 -> 1/2 -> 1/4 -> 1/8 beat on {out.id} in its last bar; {inc.id} drops at bar {out_end / BAR:g}")
    if trim_out:
        out.length_beats = min(out.length_beats, out_end)
    for c in (out, inc):
        for l in LANE_NAMES:
            c.lanes[l].points.sort(key=lambda p: p[0])
    return {"style": style, "incoming_enters_at_bar_of_outgoing": at, "bars": bars, "tempo_bpm": round(play_bpm, 3),
            "outgoing_clip_bars": out.length_beats / BAR, "notes": notes}
