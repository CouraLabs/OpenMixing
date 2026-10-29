from __future__ import annotations

from pathlib import Path

from ...core.units import BAR
from ...model.clip import Clip
from ...model.events import new_id
from ..app import Pos, tool
from ..state import LOCK, at_of, clip_of, cur, info, lib, save, span
from ..views import clip_row


def need_stems(e: dict, wanted: bool) -> None:
    if wanted and not e.get("stems"):
        if e.get("stems_legacy"):
            raise ValueError(f"{e['file']} only has the old 3-stem layout: "
                             "call separate_stems(force=True) to re-split into vocals / drums / bass / other")
        raise ValueError(f"{e['file']} has no stems yet: call separate_stems first")


def _place(c: Clip, start_at_sec, over_clip, at_bar: float | str | None, offset_ms):
    if over_clip is not None:
        if at_bar is None:
            raise ValueError("over_clip needs at_bar (a bar of that clip's playback)")
        o = clip_of(over_clip)
        at_bar = at_of(o, at_bar)
        if at_bar * BAR > o.length_beats + 1e-9:
            raise ValueError(f"at_bar {at_bar} is beyond clip {o.id} ({o.length_beats / BAR:g} bars)")
        c.start = {"anchor": o.id, "anchor_beat": at_bar * BAR}
    elif start_at_sec is not None:
        c.start = {"at_sec": float(start_at_sec)}
    if offset_ms is not None:
        c.start["offset_ms"] = float(offset_ms)


@tool()
def add_clip(track: str, src_start_bar: float | str = 0.0, bars: float | str | None = None, start_at_sec: float | None = None,
             over_clip: str | None = None, at_bar: Pos | None = None, bpm: float | None = None, label: str = "",
             gain_db: float = 0.0, key_lock: bool = True, semitones: float = 0.0, stems: bool = False) -> dict:
    """Load a track on a new deck and put it on the timeline.

    src_start_bar: where playback starts. A bar of the track (0 = first downbeat), or any position string: "64 beats", "95.5s",
    "cue:drop", "section:breakdown+4". bars: how long it plays (same formats; default: to the end of the track). Placement: start_at_sec, or
    over_clip + at_bar (anchor to a
    bar of another clip); with neither the clip goes right after the last clip (or at 0). bpm: play at this tempo (default
    native). key_lock=False makes pitch follow tempo (varispeed); semitones transposes. stems=True plays the separated stems
    (vocals / drums / bass / other, see separate_stems) so each can be mixed on its own with set_stem."""
    with LOCK:
        e = info(track)
        need_stems(e, stems)
        src_start_bar = lib().to_bars(track, src_start_bar, "src_start_bar")
        bars = None if bars is None else lib().to_bars(track, bars, "bars")
        n_bars = e["n_bars"]
        L = (n_bars - src_start_bar) if bars is None else bars
        if L <= 0:
            raise ValueError(f"src_start_bar {src_start_bar} is past the end of the track ({n_bars} bars)")
        c = Clip(new_id("c"), e["id"], src_start_bar * BAR, L * BAR, label=label or Path(e["file"]).stem, gain_db=gain_db,
                 key_lock=key_lock, semitones=semitones, stems=stems)
        if over_clip is None and start_at_sec is None:
            c.start = {"at_sec": max((span(x)[1] for x in cur().clips), default=0.0)}
        _place(c, start_at_sec, over_clip, at_bar, None)
        if bpm:
            c.tempo.points = [[0.0, bpm]]
        cur().clips.append(c)
        save()
    return clip_row(c)


@tool()
def edit_clip(clip: str, src_start_bar: float | str | None = None, bars: float | str | None = None, start_at_sec: float | None = None,
              over_clip: str | None = None, at_bar: Pos | None = None, offset_ms: float | None = None, gain_db: float | None = None,
              key_lock: bool | None = None, semitones: float | None = None, label: str | None = None,
              stems: bool | None = None) -> dict:
    """Change a clip's source start / length / placement / gain / pitch. offset_ms nudges its timing for fine phase alignment.
    stems switches the deck between the full mix and its separated stems (needs separate_stems first)."""
    with LOCK:
        c = clip_of(clip)
        need_stems(info(c.track), bool(stems))
        if src_start_bar is not None:
            c.src_start_beat = lib().to_bars(c.track, src_start_bar, "src_start_bar") * BAR
        if bars is not None:
            c.length_beats = lib().to_bars(c.track, bars, "bars") * BAR
        _place(c, start_at_sec, over_clip, at_bar, offset_ms)
        for k, v in (("gain_db", gain_db), ("key_lock", key_lock), ("semitones", semitones), ("label", label), ("stems", stems)):
            if v is not None:
                setattr(c, k, v)
        save()
    return clip_row(c)


@tool()
def remove_clip(clip: str) -> dict:
    """Delete a clip (clips anchored to it must be re-placed)."""
    with LOCK:
        c = clip_of(clip)
        dep = [x.id for x in cur().clips if x.start.get("anchor") == c.id]
        if dep:
            raise ValueError(f"clips {dep} are anchored to {clip}; re-place them first (edit_clip)")
        cur().clips.remove(c)
        cur().ducks = [d for d in cur().ducks if clip not in (d.target, d.key)]
        save()
    return {"removed": clip}
