from __future__ import annotations

from ... import mixing
from ...core.units import BAR
from ...model.curves import write
from ...model.tempo import TempoMap
from ..app import Curve, Len, Pos, tool
from ..state import LOCK, at_of, clip_of, conv_beat, info, len_of, save
from ..views import describe_events


def seed_tempo(c) -> None:
    if not c.tempo.points:
        c.tempo.points = [[0.0, info(c.track)["bpm"]]]


@tool()
def set_tempo(clip: str, at_bar: Pos, bpm: float, ramp_bars: Len = 0.0, curve: Curve = "linear", exponent: float = 2.0) -> dict:
    """Change the deck's playback tempo at a bar, optionally ramping (pitch-fader move) along a curve (linear, equal_power, log, s_curve,
    power with `exponent`, sharp_cut). Key-locked unless the clip is varispeed."""
    with LOCK:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        seed_tempo(c)
        write(c.tempo, at_bar * BAR, bpm, ramp_bars * BAR, curve, exponent)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "tempo" in e]}


@tool()
def sync_clip(clip: str, to_clip: str, at_bar: Pos = 0.0, ramp_bars: Len = 0.0, ramp_curve: Curve = "linear") -> dict:
    """Set `clip` to the tempo `to_clip` is playing at the same moment (Serato SYNC). at_bar is a bar of `clip`; the ramp follows ramp_curve."""
    with LOCK:
        c, o = clip_of(clip), clip_of(to_clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        ob = conv_beat(c, at_bar * BAR, o)
        bpm = float(TempoMap(o, info(o.track)["bpm"]).bpm_at(max(ob, 0)))
        ratio = mixing.tempo_match(bpm, info(c.track)["bpm"])["ratio_used"]
        seed_tempo(c)
        write(c.tempo, at_bar * BAR, bpm / ratio, ramp_bars * BAR, ramp_curve)
        save()
    return {"clip": clip, "bpm": round(bpm / ratio, 3)}
