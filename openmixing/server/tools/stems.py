from __future__ import annotations

import asyncio
from typing import Literal

from mcp.server.fastmcp import Context

from ...analysis import stems as stemlib
from ...core.units import BAR
from ...model.lane import STEM_LANES
from ..app import Len, Pos, tool
from ..edit import automate
from ..state import LOCK, at_of, clip_of, info, len_of, lib, save
from ..views import describe_events, jsonable
from .clips import need_stems


@tool()
async def separate_stems(tracks: list[str], force: bool = False, ctx: Context = None) -> dict:
    """Split tracks into four stems with Demucs (htdemucs) and cache them in .openmixing/stems/: vocals, drums, bass and other
    (synths, pads, keys, guitars). Takes roughly 0.5-2 min per track the first time (the model downloads once). Afterwards use
    add_clip / edit_clip with stems=True, then mix the parts with set_stem or stem_preset. Tracks that only have the old 3-stem
    layout are re-split automatically. Each track comes back with a quality report: per-stem energy share, bleed and confidence."""
    if not stemlib.available():
        raise RuntimeError("stem separation needs Demucs: pip install 'openmixing[stems]'  (installs torch + demucs)")
    done, errors = [], {}
    for i, ref in enumerate(tracks):
        if ctx:
            await ctx.report_progress(i, len(tracks), f"separating {ref}")
        try:
            done.append(await asyncio.to_thread(lib().separate, ref, force))
        except Exception as e:
            errors[ref] = str(e)
    if ctx:
        await ctx.report_progress(len(tracks), len(tracks), "done")
    return jsonable({"stems": list(stemlib.STEMS), "separated": [{"track": r["track"], "cached": r["cached"],
                                                                  "quality": r.get("quality")} for r in done], "errors": errors})


STEM_PRESETS = {
    "instrumental": {"vocals": 0.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
    "acapella": {"vocals": 1.0, "drums": 0.0, "bass": 0.0, "other": 0.0},
    "drumless": {"vocals": 1.0, "drums": 0.0, "bass": 1.0, "other": 1.0},
    "bassless": {"vocals": 1.0, "drums": 1.0, "bass": 0.0, "other": 1.0},
    "full": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
}


def _need_stem_clip(c) -> None:
    if not c.stems:
        raise ValueError(f"clip {c.id} plays the full mix: call edit_clip(stems=True) first")


@tool()
def set_stem(clip: str, stem: Literal["vocals", "drums", "bass", "other", "all"], at_bar: Pos, level: float = 1.0,
             ramp_bars: Len = 0.0) -> dict:
    """Stem fader on a clip that plays stems (add_clip / edit_clip stems=True): 0 = muted, 1 = unity, up to 1.5. ramp_bars = 0 snaps.
    E.g. vocals of one track over the drums + bass + other of another (a mashup), drop the drums for a breakdown, or bring the
    vocals in on a phrase. Stems feed the clip's EQ / filter / fader / fx, so those still work on top."""
    if not 0.0 <= level <= 1.5:
        raise ValueError("level must be in [0, 1.5]")
    names = list(STEM_LANES) if stem == "all" else [stem]
    return automate(clip, at_bar, ramp_bars, {STEM_LANES[n]: level for n in names}, "stem_", _need_stem_clip)


@tool()
def stem_preset(clip: str, preset: Literal["instrumental", "acapella", "drumless", "bassless", "full"], at_bar: Pos = 0.0,
                ramp_bars: Len = 0.0) -> dict:
    """Set all four stem faders at once from `at_bar`: instrumental (no vocals), acapella (vocals only), drumless, bassless or full.
    Turns the clip's stems on (needs separate_stems first). ramp_bars fades between the previous state and the preset."""
    with LOCK:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        if not c.stems:
            need_stems(info(c.track), True)
            c.stems = True
        for n, level in STEM_PRESETS[preset].items():
            c.lanes[STEM_LANES[n]].set(at_bar * BAR, level, ramp_bars * BAR)
        save()
    return {"clip": clip, "preset": preset, "events": [e for e in describe_events(c) if "stem_" in e]}
