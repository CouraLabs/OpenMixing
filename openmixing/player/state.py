from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engine.segments import segments
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap, resolve_starts
from .phrases import phrase_beats
from .waveform import band_envelope, clip_waveform, track_features

META_KEYS = ("start_sec", "end_sec", "duration_sec", "integrated_lufs", "loudness_range_lu", "true_peak_dbtp", "peaks", "peak_block_sec", "format")


def _waveform(c, e: dict, tm: TempoMap, feats: dict) -> dict | None:
    try:
        if c.track not in feats:
            feats[c.track] = track_features(e["path"])
        return clip_waveform(c, e, tm, feats[c.track])
    except Exception:
        return None


def _clip_state(c, e: dict, d: dict, tm: TempoMap, t0: float, wf: dict | None) -> dict:
    sec = lambda b: round(t0 + float(tm.beat_to_sec(b)), 3)
    clamp = lambda b: sec(max(min(b, c.length_beats), 0))
    lane = lambda pts: [[sec(b), v] for b, v in pts]
    beats = (t0 + tm.beat_to_sec(np.append(np.arange(0, c.length_beats, 1.0), c.length_beats))).round(3).tolist()
    segs = [[round(g[1], 4), round(g[2], 4), round(g[3], 4), g[4] if g[0] == "loop" else 0] for g in segments(c)]
    phrases = [{"kind": p["kind"], "start": sec(p["beat"]), "end": sec(p["end_beat"])}
               for p in phrase_beats(d["structure"], c.src_start_beat, c.length_beats)]
    return {
        "id": c.id, "label": c.label, "file": e["file"], "title": d["tags"].get("title"), "artist": d["tags"].get("artist"),
        "native_bpm": round(e["bpm"], 2), "camelot": e["camelot"], "key": e["key"], "energy": e["energy"], "wf": wf,
        "start": clamp(0), "end": clamp(c.length_beats), "src_start_beat": c.src_start_beat, "length_beats": c.length_beats,
        "beats": beats, "segs": segs, "phrases": phrases,
        "tempo": lane(c.tempo.points or [[0, e["bpm"]]]),
        "lanes": {k: lane(l.points) for k, l in c.lanes.items() if l.points},
        "fx": [{"id": f.id, "fx": f.fx, "start": sec(f.beat), "end": clamp(f.beat + f.duration_beats), "beats": f.beats, "wet": f.wet} for f in c.fx],
        "loops": [{"id": l.id, "start": sec(l.beat), "end": clamp(l.beat + l.length_beats), "beat": l.beat, "length_beats": l.length_beats,
                   "size_beats": l.size_beats, "passes": int(np.ceil(l.length_beats / l.size_beats - 1e-9)), "roll": l.roll} for l in c.loops],
        "section_map": [{"label": s["label"], "start": s["start_sec"], "end": s["end_sec"]} for s in d["structure"]["sections"]],
    }


def build_state(lib, model: SetModel, meta: dict) -> dict:
    starts = resolve_starts(model, lambda t: lib.effective(t)["bpm"])
    clips, feats = [], {}
    for c in model.clips:
        e = lib.effective(c.track)
        tm = TempoMap(c, e["bpm"])
        clips.append(_clip_state(c, e, lib.analyze(c.track), tm, starts[c.id], _waveform(c, e, tm, feats)))
    try:
        bands = band_envelope(meta["path"], meta["peak_block_sec"])
    except Exception:
        bands = None
    return {"bands": bands, "set": model.name, "version": model.version, "audio": Path(meta["path"]).name, "clips": clips,
            **{k: meta.get(k) for k in META_KEYS}}
