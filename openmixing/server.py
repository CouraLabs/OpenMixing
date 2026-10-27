"""OpenMixing MCP server (stdio). Open your agent in the tracks folder and it can DJ."""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from . import tags as taglib, tracklist as tracklistlib, candidates, plan as planlib, planner, render as renderer, stems as stemlib, validate
from .analysis.features import harmonic_neighbours
from .errors import OpenMixingError, structured
from .units import duration_bars
from . import region as regionlib
from .versions import versions
from .dsp import fx as fxm
from .guide import GUIDE
from .library import Library
from .models import BEAT_DIVISIONS, BEAT_LABELS, SR, parse_beats
from .setmodel import LANES, STEM_LANES, Clip, FxEvent, Lane, LoopEvent, SetModel, TempoMap, new_id, resolve_starts

BAR = 4.0
mcp = FastMCP("openmixing", instructions=GUIDE)


def tool():
    """mcp.tool() whose exceptions come back as structured {code, field, message, fix} errors."""
    def deco(fn):
        mcp.tool()(structured(fn))
        return fn  # direct Python callers (tests, other modules) still see the plain exceptions
    return deco

_lock = threading.RLock()
_state: dict = {"lib": None, "set": None, "ui": None}

Beats = Annotated[str | float, Field(description="Tempo-sync multiplier in beats: one of 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32")]


# ============================================================== helpers
def lib() -> Library:
    if _state["lib"] is None:
        _state["lib"] = Library()
    return _state["lib"]


def cur() -> SetModel:
    if _state["set"] is None:
        names = lib().list_sets()
        last = lib().state / "current_set"
        want = last.read_text().strip() if last.exists() else None
        if want in names:
            _state["set"] = lib().load_set(want)
        else:
            _state["set"] = SetModel("main")
            lib().save_set(_state["set"])
    return _state["set"]


def save() -> None:
    m = cur()
    lib().save_set(m)
    (lib().state / "current_set").write_text(m.name)


def clip_of(cid: str) -> Clip:
    return cur().clip(cid)


def info(track_id: str) -> dict:
    return lib().render_info(track_id)


def span(c: Clip) -> tuple[float, float]:
    st = resolve_starts(cur(), lambda t: info(t)["bpm"])[c.id]
    return st, st + float(TempoMap(c, info(c.track)["bpm"]).beat_to_sec(c.length_beats))


def conv_beat(src: Clip, beat: float, dst: Clip) -> float:
    """Beat of `src` expressed as a beat of `dst` (via the set timeline)."""
    starts = resolve_starts(cur(), lambda t: info(t)["bpm"])
    t = starts[src.id] + float(TempoMap(src, info(src.track)["bpm"]).beat_to_sec(beat))
    return float(TempoMap(dst, info(dst.track)["bpm"]).sec_to_beat(t - starts[dst.id]))


Pos = Annotated[float | str, Field(description="Position in the clip's playback: output bars (0 = clip start), \"32 beats\", or a place in the "
                                              "source track (\"cue:drop\", \"section:breakdown+4\", \"95.5s\"), converted with the clip's src_start_bar")]
Len = Annotated[float | str, Field(description="Length in bars, or \"N beats\"")]


def at_of(c: Clip, v, field: str = "at_bar") -> float:
    return lib().out_bars(c, v, field)


def len_of(v, field: str = "bars") -> float:
    return duration_bars(v, field)


def parse_beats_arg(v) -> float:
    try:
        return parse_beats(v)
    except ValueError as e:
        raise ValueError(str(e)) from None


def clip_row(c: Clip) -> dict:
    e = info(c.track)
    s, t = span(c)
    return {"clip": c.id, "label": c.label, "track": e["file"], "start_sec": round(s, 2), "end_sec": round(t, 2),
            "src_start_bar": c.src_start_beat / BAR, "bars": c.length_beats / BAR, "bpm_native": round(e["bpm"], 2),
            "camelot": e["camelot"], "stems": c.stems, "start": c.start}


def describe_events(c: Clip) -> list[str]:
    ev = []
    for name, lane in c.lanes.items():
        pts = lane.points
        for i, (b, v) in enumerate(pts):
            if i + 1 < len(pts) and abs(pts[i + 1][1] - v) < 1e-9 and (i == 0 or abs(pts[i - 1][1] - v) < 1e-9):
                continue  # inside a flat run
            if i == 0 and abs(v - lane.neutral) < 1e-9 and (len(pts) < 2 or pts[1][0] - b < 0.03):
                continue
            ev.append((b, f"{name} = {v:g}"))
    for i, (b, v) in enumerate(c.tempo.points):
        ev.append((b, f"tempo = {v:.2f} bpm"))
    for f in c.fx:
        ev.append((f.beat, f"fx {f.id}: {f.fx}{' on ' + f.stem + ' only' if f.stem else ''} {BEAT_LABELS.get(f.beats, f.beats)} beats, wet {f.wet:g}, for {f.duration_beats / BAR:g} bars {f.params or ''}".strip()))
    for l in c.loops:
        ev.append((l.beat, f"loop {l.id}: {BEAT_LABELS.get(l.size_beats, l.size_beats)}-beat {'roll ' if l.roll else ''}for {l.length_beats / BAR:g} bars"))
    return [f"bar {b / BAR:g}: {t}" for b, t in sorted(ev)]


def lane_name(band: str) -> str:
    return {"low": "eq_low", "mid": "eq_mid", "high": "eq_high"}[band]


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# ============================================================== library & analysis
@tool()
def list_tracks(analyzed_only: bool = False) -> dict:
    """List the audio files in the tracks folder (the working directory) with cached bpm / key / energy when analyzed."""
    rows = []
    for t in lib().scan():
        d = lib().cached(t)
        if analyzed_only and not d:
            continue
        row = {"id": t["id"], "file": t["file"], "analyzed": bool(d), "stems": bool(lib().stem_paths(t))}
        if t.get("duplicate_of"):
            row["duplicate_of"] = t["duplicate_of"]
        ut = lib().user_tags(t["id"])
        row.update({k: ut[k] for k in ("rating", "color", "tags") if k in ut})
        if d:
            tg = lib().merged_tags(t["id"], d["tags"])
            fi = d.get("file_info", {})
            row.update(format=fi.get("codec"), sample_rate=fi.get("sample_rate"), bit_depth=fi.get("bit_depth"), channels=fi.get("channels"))
            row.update(title=tg.get("title"), artist=tg.get("artist"), bpm=d["tempo"]["bpm"], camelot=d["key"]["camelot"],
                       energy=d["energy"]["level_1_10"], duration_sec=d["duration_sec"], lufs=d["loudness"]["integrated_lufs"],
                       manual_overrides=bool(lib().overrides().get(t["id"])))
        rows.append(row)
    return {"folder": str(lib().root), "count": len(rows), "tracks": rows}


@tool()
async def analyze_track(track: str, force: bool = False, include_beats: bool = False, include_waveform: bool = False,
                        region: list[float | str] | None = None, ctx: Context = None) -> dict:
    """Full analysis of one track (cached in .openmixing/). `track` is an id, file name, or unique part of the name.

    Returns: file info (codec, bitrate, ...), embedded tags, tempo (bpm, first downbeat, grid stability, drift),
    key (name, Camelot, confidence, candidates, key over time, compatible keys), loudness (LUFS, LRA, true peak,
    clipping, gain to -14 LUFS, leading/trailing silence), stereo, spectrum (band shares, centroid, MFCC, contrast,
    cutoff / transcode check), energy (1-10, danceability heuristic), structure (sections with labels, phrase length,
    per-bar energy and bass curves), mix points (cues, mix_in, mix_out), loop candidates, manual overrides, and cached
    online metadata if `lookup_online` was run. Takes ~10-30 s the first time. Every estimate carries `confidence` (0..1) and
    `confidence_components` (what it was derived from).

    region=[start, end] (any position format, e.g. [32, "section:drop+8"]) replaces the per-bar arrays, sections and loop candidates of
    `structure` with just that range, and trims key_over_time; use get_region for more fields.
    """
    d = await asyncio.to_thread(lib().analyze, track, force)
    out = dict(d)
    out["versions"] = versions()
    eff = lib().effective(track)
    out["effective"] = {k: eff[k] for k in ("bpm", "first_downbeat", "gain_db", "camelot", "key", "overrides")}
    if (o := lib().cached_online(d["id"])):
        out["online"] = o
    t = d["tempo"]
    if include_beats:
        fd, p = eff["first_downbeat"], t["beat_period_sec"] * t["bpm"] / eff["bpm"]
        out["beat_times_sec"] = [round(fd + k * p, 3) for k in range(int((d["duration_sec"] - fd) / p))]
    if include_waveform:
        out["waveform"] = lib().waveform(d["id"])
    if region is not None:
        a, b = _range(d["id"], region)
        keep = {k: v for k, v in out["structure"].items() if k not in ("sections", "energy_curve", "bass_curve", "onset_curve", "loop_candidates")}
        keep["region"] = regionlib.slice_region(d, eff, a, b, ("energy", "bass", "onsets", "sections", "loops"))
        out["structure"] = keep
        out["key"] = {**out["key"], "key_over_time": [k for k in out["key"]["key_over_time"] if keep["region"]["start_sec"] - 1e-9 <= k["start_sec"] < keep["region"]["end_sec"]]}
        out["region"] = [a, b]
    return jsonable(out)


def _range(track: str, rng) -> tuple[float, float]:
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise OpenMixingError("invalid_value", f"region must be [start, end], got {rng!r}", "region", "e.g. [32, 64] or [\"cue:drop\", \"section:breakdown\"]")
    a, b = (lib().to_bars(track, v, "region") for v in rng)
    if b <= a:
        raise OpenMixingError("invalid_range", f"region end ({b:g} bars) is not after start ({a:g} bars)", "region", "give end > start")
    return a, b


@tool()
def get_region(track: str, start: float | str = 0, end: float | str | None = None,
               what: list[str] | None = None, max_points: int = 128) -> dict:
    """Detail for one range of a track instead of the whole analysis. start / end are any position (bars, "64 beats", "95.5s",
    "cue:<name>", "section:<label>+N"); end defaults to start + 16 bars. `what` picks fields from: energy, bass, onsets (per-bar curves),
    vocal (needs stems), beats, sections, loops, cues, waveform; default energy, bass, sections. Per-bar arrays are averaged down to
    at most `max_points` values (`step_bars` says by how much)."""
    e = lib().effective(track)
    d = lib().analyze(e["id"])
    a = lib().to_bars(e["id"], start, "start")
    b = a + 16 if end is None else lib().to_bars(e["id"], end, "end")
    if b <= a:
        raise OpenMixingError("invalid_range", f"end ({b:g} bars) is not after start ({a:g} bars)", "end", "give end > start")
    if max_points < 1:
        raise OpenMixingError("invalid_value", "max_points must be at least 1", "max_points", "use 16-512")
    what = regionlib.check_what(what or regionlib.DEFAULT_WHAT)
    vocal = None
    if "vocal" in what and e.get("stems"):
        n_bars = int(d["tempo"]["n_bars"]) if "bpm" not in e["overrides"] else int((e["duration"] - e["first_downbeat"]) * e["bpm"] / 240)
        vocal = candidates.vocal_entry(lib(), e, n_bars)
    r = regionlib.slice_region(d, e, a, b, what, max_points, vocal, lib().cues(e["id"]), lib().waveform(e["id"]) if "waveform" in what else None)
    return jsonable({"track": e["file"], "id": e["id"], "duration_sec": d["duration_sec"], **r, "versions": versions()})


def _analyze_worker(root: str, ref: str, force: bool) -> str:
    Library(root).analyze(ref, force)
    return ref


@tool()
async def analyze_all(force: bool = False, ctx: Context = None) -> dict:
    """Analyze every track in the folder that is not cached yet (parallel). Reports progress."""
    todo = [t for t in lib().scan() if force or not lib().cached(t)]
    done, errors = 0, {}
    if todo:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1))) as pool:
            async def run(t):
                try:
                    await loop.run_in_executor(pool, _analyze_worker, str(lib().root), t["id"], force)
                    return t, None
                except Exception as e:
                    return t, e

            for f in asyncio.as_completed([run(t) for t in todo]):
                t, err = await f
                if err:
                    errors[t["file"]] = str(err)
                done += 1
                if ctx:
                    await ctx.report_progress(done, len(todo), t["file"])
    return {"analyzed": len(todo) - len(errors), "skipped_cached": len(lib().scan()) - len(todo), "errors": errors}


@tool()
async def lookup_online(track: str, providers: list[str] | None = None, artist: str | None = None, title: str | None = None,
                        refresh: bool = False) -> dict:
    """Look the track up online and cache the result: MusicBrainz (ids, genres, tags, release), Deezer (bpm, gain, rank,
    label, genres), iTunes (genre, release date, artwork), ReccoBeats (Spotify-style audio features by ISRC:
    danceability, energy, valence, acousticness, tempo ...). Key-gated, enabled by env vars: Last.fm (LASTFM_API_KEY),
    Discogs (DISCOGS_TOKEN), AcoustID fingerprinting (ACOUSTID_API_KEY + chromaprint's fpcalc). Artist / title come
    from tags or the "Artist - Title" file name; pass them to override. `consensus` merges genres, ISRC, dates and
    cross-checks the reference bpm against the analysis."""
    return jsonable(await asyncio.to_thread(lib().online, track, providers, artist, title, refresh))


@tool()
def set_cue(track: str, name: str, position: float | str, color: str | None = None, note: str = "") -> dict:
    """Create or move a named hot cue on a track (persisted). `position` is any position format: bars, "64 beats", "95.5s",
    "section:drop+4". Cues can then be used anywhere a position is accepted as "cue:<name>"."""
    t = lib().resolve(track)
    bar = lib().to_bars(t["id"], position, "position")
    return {"track": t["file"], "cues": lib().set_cue(t["id"], name, bar, color, note)}


@tool()
def list_cues(track: str) -> dict:
    """Hot cues you set on a track plus auto-suggested cues from analysis (first downbeat, drops, breakdowns, mix in/out)."""
    t = lib().resolve(track)
    d = lib().analyze(t["id"])
    return jsonable({"track": t["file"], "cues": lib().cues(t["id"]), "suggested": d["mix_points"]["cue_points"],
                     "mix_in": d["mix_points"]["mix_in"], "mix_out": d["mix_points"]["mix_out"]})


@tool()
def delete_cue(track: str, name: str) -> dict:
    """Remove a hot cue."""
    t = lib().resolve(track)
    return {"track": t["file"], "cues": lib().delete_cue(t["id"], name)}


@tool()
def override_track(track: str, bpm: float | None = None, first_downbeat_sec: float | None = None,
                   shift_downbeat_beats: float | None = None, camelot: str | None = None, key: str | None = None,
                   gain_db: float | None = None, clear: list[str] | None = None) -> dict:
    """Correct the analysis (persisted). bpm / first_downbeat_sec fix the beat grid, shift_downbeat_beats moves the
    downbeat by whole beats (e.g. 1 when the grid is a beat late), camelot / key fix the key, gain_db sets the level
    trim. `clear` lists override names to remove (bpm, first_downbeat, camelot, key, gain_db)."""
    t = lib().resolve(track)
    d = lib().analyze(t["id"])
    kv: dict = {k: None for k in (clear or [])}
    eff = lib().effective(t["id"])
    if bpm is not None:
        kv["bpm"] = bpm
    if first_downbeat_sec is not None:
        kv["first_downbeat"] = first_downbeat_sec
    if shift_downbeat_beats is not None:
        kv["first_downbeat"] = eff["first_downbeat"] + shift_downbeat_beats * 60.0 / eff["bpm"]
    for k, v in (("camelot", camelot), ("key", key), ("gain_db", gain_db)):
        if v is not None:
            kv[k] = v
    return {"track": t["file"], "overrides": lib().set_override(t["id"], **kv), "note": "clips in existing sets use the corrected grid on the next render"}


@tool()
def find_tracks(query: str | None = None, bpm_min: float | None = None, bpm_max: float | None = None, camelot: str | None = None,
                compatible_with_camelot: str | None = None, energy_min: int | None = None, energy_max: int | None = None,
                genre: str | None = None, rating_min: int | None = None, color: str | None = None, tag: str | None = None,
                sort_by: Literal["bpm", "energy", "key", "name", "duration"] = "name") -> dict:
    """Filter analyzed tracks. `rating_min` / `color` / `tag` match the tags set with set_tags. `compatible_with_camelot` (e.g. "8A") keeps same / adjacent / relative keys. `genre` matches
    tags and cached online genres."""
    ok_keys = None
    if compatible_with_camelot:
        n = harmonic_neighbours(compatible_with_camelot)
        ok_keys = {k for v in n.values() for k in v}
    rows = []
    for t in lib().scan():
        if not lib().cached(t):
            continue
        e, d = lib().effective(t["id"]), lib().analyze(t["id"])
        on = lib().cached_online(t["id"]) or {}
        gs = " ".join([str(d["tags"].get("genre", "")), *(on.get("consensus", {}).get("genres") or [])]).lower()
        if query and query.lower() not in (t["file"] + str(d["tags"])).lower():
            continue
        if (bpm_min and e["bpm"] < bpm_min) or (bpm_max and e["bpm"] > bpm_max) or (camelot and e["camelot"] != camelot):
            continue
        if (ok_keys and e["camelot"] not in ok_keys) or (energy_min and e["energy"] < energy_min) or (energy_max and e["energy"] > energy_max):
            continue
        if genre and genre.lower() not in gs:
            continue
        ut = lib().user_tags(t["id"])
        if (rating_min and ut.get("rating", 0) < rating_min) or (color and ut.get("color") != color.lower()) or (tag and tag.strip() not in ut.get("tags", [])):
            continue
        rows.append({"id": t["id"], "file": t["file"], "bpm": round(e["bpm"], 2), "camelot": e["camelot"], "energy": e["energy"], "duration_sec": e["duration"],
                     **{k: ut[k] for k in ("rating", "color", "tags") if k in ut}})
    keyf = {"bpm": lambda r: r["bpm"], "energy": lambda r: r["energy"], "key": lambda r: (int(r["camelot"][:-1]), r["camelot"][-1]),
            "name": lambda r: r["file"].lower(), "duration": lambda r: r["duration_sec"]}[sort_by]
    return {"count": len(rows), "tracks": sorted(rows, key=keyf)}


@tool()
def set_tags(track: str, title: str | None = None, artist: str | None = None, album: str | None = None, genre: str | None = None,
             year: int | None = None, comment: str | None = None, label: str | None = None, rating: int | None = None,
             color: str | None = None, isrc: str | None = None, tags: list[str] | None = None, notes: str | None = None,
             clear: list[str] | None = None, write_to_file: bool = False, backup: bool = True) -> dict:
    """Set track metadata. By default it is stored only in .openmixing/tags.json (your audio file is not touched) and overlays the file
    tags in list_tracks / tracklists. write_to_file=True also writes title / artist / album / genre / year / comment / label / isrc into
    the file (MP3, WAV, AIFF via ID3; FLAC, OGG, Opus via Vorbis comments; M4A via MP4 atoms), after copying the original to
    .openmixing/backups/ unless backup=False. rating (0-5), color (red, orange, yellow, green, aqua, blue, purple, pink or #rrggbb),
    tags (free labels, replaced as a list) and notes always stay in the sidecar. `clear` lists fields to remove from the sidecar."""
    with _lock:
        return jsonable(lib().set_user_tags(track, dict(title=title, artist=artist, album=album, genre=genre, year=year, comment=comment,
                                                       label=label, rating=rating, color=color, isrc=isrc, tags=tags, notes=notes),
                                            write_to_file, backup, clear))


@tool()
def check_files() -> dict:
    """Format report for every audio file in the tracks folder: format, codec, sample rate, bit depth, channels, duration, whether it is
    lossless, whether tags can be written into it, and warnings (resampling, downmix, low bitrate). Corrupt or unsupported files are listed
    with an error code and a fix instead of failing the whole call. Files with audio extensions OpenMixing cannot read are listed as ignored."""
    from .analysis.meta import UNSUPPORTED_AUDIO_EXTS, format_report
    from .errors import classify
    rows = []
    for t in lib().scan():
        try:
            rows.append({"id": t["id"], "file": t["file"], **format_report(t["path"])})
        except Exception as e:
            rows.append({"id": t["id"], "file": t["file"], "supported": False, **classify(e).payload()["error"]})
    ignored = sorted(str(p.relative_to(lib().root)) for p in lib().root.rglob("*")
                     if p.suffix.lower() in UNSUPPORTED_AUDIO_EXTS and ".openmixing" not in p.parts)
    return {"count": len(rows), "unusable": sum(1 for r in rows if not r["supported"]),
            "with_warnings": sum(1 for r in rows if r.get("warnings")), "files": rows, "ignored_audio_files": ignored}


@tool()
def export_tracklist(format: Literal["text", "json", "cue"] = "text", set_name: str | None = None, offset_sec: float | None = None) -> dict:
    """Tracklist / cue sheet of the set: start time of every clip in the rendered file (time 0 = the earliest clip, or offset_sec) with
    artist, title, bpm and key from the track tags. Written to .openmixing/renders/<set>.tracklist.txt|json or <set>.cue next to the render."""
    m = lib().load_set(set_name) if set_name else cur()
    if not m.clips:
        raise ValueError("the set has no clips")
    tl = renderer.timeline(m, info)
    rs = tracklistlib.rows(tl, info, lambda tid: lib().merged_tags(info(tid)["id"], lib().analyze(tid)["tags"]), offset_sec)
    audio = next((f"{lib().safe_name(m.name)}.{x}" for x in ("flac", "wav", "mp3") if lib().render_path(m.name, x).exists()), f"{lib().safe_name(m.name)}.flac")
    ext, body = {"text": ("tracklist.txt", tracklistlib.as_text(m.name, rs)), "json": ("tracklist.json", tracklistlib.as_json(m.name, rs, audio)),
                 "cue": ("cue", tracklistlib.as_cue(m.name, rs, audio))}[format]
    path = lib().output_path(lib().state / "renders" / f"{lib().safe_name(m.name)}.{ext}", "set_name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return {"path": str(path), "format": format, "tracks": len(rs), "audio_file": audio, "content": body if format != "json" else rs}


@tool()
def compare_tracks(from_track: str, to_track: str) -> dict:
    """Mixability of from_track -> to_track: key relation, tempo stretch needed (half/double time aware), energy delta, score 0-100,
    and a suggested transition style / length."""
    a, b = lib().effective(from_track), lib().effective(to_track)
    c = planner.compat(a, b)
    k, tp = c["key"]["distance"], abs(c["tempo"]["stretch_percent"])
    if tp > 8:
        sug = {"style": "echo_out", "bars": 4, "why": "tempos too far apart to beatmatch cleanly"}
    elif k <= 1 and tp <= 4:
        sug = {"style": "long_blend", "bars": 32, "why": "harmonic match and close tempo: blend long"}
    elif k <= 2:
        sug = {"style": "bass_swap", "bars": 16, "why": "compatible keys: classic bass-swap blend"}
    else:
        sug = {"style": "filter_fade", "bars": 8, "why": "keys clash: keep the overlap short and filtered"}
    return {"from": a["file"], "to": b["file"], **c, "suggested_transition": sug}


@tool()
def suggest_next(track: str, count: int = 8, max_stretch_percent: float = 8.0, min_score: float = 0.0) -> dict:
    """Rank analyzed tracks to follow `track` (harmonic + tempo + energy)."""
    a = lib().effective(track)
    rows = []
    for t in lib().scan():
        if t["id"] == a["id"] or not lib().cached(t):
            continue
        b = lib().effective(t["id"])
        c = planner.compat(a, b)
        if abs(c["tempo"]["stretch_percent"]) <= max_stretch_percent and c["score"] >= min_score:
            rows.append({"id": b["id"], "file": b["file"], "score": c["score"], "key": f"{b['camelot']} ({c['key']['relation']})",
                         "bpm": round(b["bpm"], 2), "stretch_percent": c["tempo"]["stretch_percent"], "energy": b["energy"]})
    return {"after": a["file"], "candidates": sorted(rows, key=lambda r: -r["score"])[:count]}


@tool()
def suggest_order(tracks: list[str] | None = None, strategy: Literal["harmonic", "energy_build", "bpm_ramp"] = "harmonic",
                  start_with: str | None = None) -> dict:
    """Order tracks (default: all analyzed) into a set: harmonic = best pairwise compatibility, energy_build = rising energy,
    bpm_ramp = rising tempo."""
    refs = tracks or [t["id"] for t in lib().scan() if lib().cached(t)]
    effs = [lib().effective(r) for r in refs]
    path = planner.order_tracks(effs, strategy, lib().resolve(start_with)["id"] if start_with else None)
    steps = [{"track": e["file"], "bpm": round(e["bpm"], 2), "camelot": e["camelot"], "energy": e["energy"]} for e in path]
    for i in range(1, len(path)):
        c = planner.compat(path[i - 1], path[i])
        steps[i]["from_previous"] = {"score": c["score"], "key": c["key"]["relation"], "stretch_percent": c["tempo"]["stretch_percent"]}
    return {"strategy": strategy, "order": steps}


# ============================================================== sets & clips
@tool()
def list_sets() -> dict:
    """Saved sets in this folder and the current one."""
    return {"sets": lib().list_sets(), "current": cur().name}


@tool()
def new_set(name: str, notes: str = "") -> dict:
    """Create (or reset) a set and make it current."""
    with _lock:
        _state["set"] = SetModel(name, notes=notes)
        save()
    return {"current": name}


@tool()
def open_set(name: str) -> dict:
    """Make a saved set current."""
    with _lock:
        _state["set"] = lib().load_set(name)
        save()
    return {"current": name, "clips": len(cur().clips)}


@tool()
def get_set(detail: bool = False) -> dict:
    """The current set: clips with resolved times, plus a readable event list per clip (detail=True adds raw automation points)."""
    m = cur()
    clips = []
    for c in sorted(m.clips, key=lambda c: span(c)[0]):
        row = clip_row(c)
        row["events"] = describe_events(c)
        if detail:
            row["raw"] = c.to_dict()
        clips.append(row)
    return jsonable({"name": m.name, "notes": m.notes, "duration_sec": max((r["end_sec"] for r in clips), default=0), "clips": clips,
                     "versions": versions()})


def _need_stems(e: dict, wanted: bool) -> None:
    if wanted and not e.get("stems"):
        if e.get("stems_legacy"):
            raise ValueError(f"{e['file']} only has the old 3-stem layout: call separate_stems(force=True) to re-split into vocals / drums / bass / other")
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
    "cue:drop", "section:breakdown+4". bars: how long it plays (same formats; default: to the end of the track). Placement: start_at_sec, or over_clip + at_bar (anchor to a
    bar of another clip); with neither the clip goes right after the last clip (or at 0). bpm: play at this tempo (default
    native). key_lock=False makes pitch follow tempo (varispeed); semitones transposes. stems=True plays the separated stems
    (vocals / drums / bass / other, see separate_stems) so each can be mixed on its own with set_stem."""
    with _lock:
        e = info(track)
        _need_stems(e, stems)
        src_start_bar = lib().to_bars(track, src_start_bar, "src_start_bar")
        bars = None if bars is None else lib().to_bars(track, bars, "bars")
        d = lib().analyze(track)
        n_bars = d["tempo"]["n_bars"] if "bpm" not in e["overrides"] else int((e["duration"] - e["first_downbeat"]) * e["bpm"] / 240)
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
    with _lock:
        c = clip_of(clip)
        _need_stems(info(c.track), bool(stems))
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
    with _lock:
        c = clip_of(clip)
        dep = [x.id for x in cur().clips if x.start.get("anchor") == c.id]
        if dep:
            raise ValueError(f"clips {dep} are anchored to {clip}; re-place them first (edit_clip)")
        cur().clips.remove(c)
        save()
    return {"removed": clip}


# ============================================================== DJ controls
@tool()
def set_tempo(clip: str, at_bar: Pos, bpm: float, ramp_bars: Len = 0.0) -> dict:
    """Change the deck's playback tempo at a bar, optionally ramping (pitch-fader move). Key-locked unless the clip is varispeed."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        if not c.tempo.points:
            c.tempo.points = [[0.0, info(c.track)["bpm"]]]
        c.tempo.set(at_bar * BAR, bpm, ramp_bars * BAR)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "tempo" in e]}


@tool()
def sync_clip(clip: str, to_clip: str, at_bar: Pos = 0.0, ramp_bars: Len = 0.0) -> dict:
    """Set `clip` to the tempo `to_clip` is playing at the same moment (Serato SYNC). at_bar is a bar of `clip`."""
    with _lock:
        c, o = clip_of(clip), clip_of(to_clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        ob = conv_beat(c, at_bar * BAR, o)
        bpm = float(TempoMap(o, info(o.track)["bpm"]).bpm_at(max(ob, 0)))
        ratio = planner.tempo_match(bpm, info(c.track)["bpm"])["ratio_used"]
        if not c.tempo.points:
            c.tempo.points = [[0.0, info(c.track)["bpm"]]]
        c.tempo.set(at_bar * BAR, bpm / ratio, ramp_bars * BAR)
        save()
    return {"clip": clip, "bpm": round(bpm / ratio, 3)}


@tool()
def set_eq(clip: str, band: Literal["low", "mid", "high", "all"], at_bar: Pos, gain_db: float | Literal["kill"] = 0.0,
           ramp_bars: Len = 0.0) -> dict:
    """Move a 3-band EQ knob on a deck (low < 200 Hz, mid 200-2500 Hz, high > 2.5 kHz). gain_db from -80 ("kill") to +6; 0 = flat.
    ramp_bars = 0 snaps (de-clicked over ~10 ms). Works on the playing deck or the incoming one."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        g = -80.0 if gain_db == "kill" else float(gain_db)
        if not -80.0 <= g <= 6.0:
            raise ValueError("gain_db must be in [-80 (kill), +6]")
        for b in (["low", "mid", "high"] if band == "all" else [band]):
            c.lanes[lane_name(b)].set(at_bar * BAR, g, ramp_bars * BAR)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "eq_" in e]}


@tool()
def set_filter(clip: str, at_bar: Pos, value: float, ramp_bars: Len = 0.0) -> dict:
    """DJ filter knob: -1 = low-pass fully closed (muffled), 0 = off, +1 = high-pass fully closed (thin). Sweep with ramp_bars."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        c.lanes["filter"].set(at_bar * BAR, float(np.clip(value, -1, 1)), ramp_bars * BAR)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "filter" in e]}


@tool()
def set_volume(clip: str, at_bar: Pos, level: float, ramp_bars: Len = 0.0) -> dict:
    """Channel fader (0-1.5, 1 = unity). It sits before the fx, so echo / reverb tails keep ringing when it closes."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        c.lanes["volume"].set(at_bar * BAR, float(np.clip(level, 0, 1.5)), ramp_bars * BAR)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "volume" in e]}


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
    return jsonable({"stems": list(stemlib.STEMS), "separated": [{"track": r["track"], "cached": r["cached"], "quality": r.get("quality")} for r in done], "errors": errors})


@tool()
def set_stem(clip: str, stem: Literal["vocals", "drums", "bass", "other", "all"], at_bar: Pos, level: float = 1.0,
             ramp_bars: Len = 0.0) -> dict:
    """Stem fader on a clip that plays stems (add_clip / edit_clip stems=True): 0 = muted, 1 = unity, up to 1.5. ramp_bars = 0 snaps.
    E.g. vocals of one track over the drums + bass + other of another (a mashup), drop the drums for a breakdown, or bring the
    vocals in on a phrase. Stems feed the clip's EQ / filter / fader / fx, so those still work on top."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        if not c.stems:
            raise ValueError(f"clip {clip} plays the full mix: call edit_clip(stems=True) first")
        if not 0.0 <= level <= 1.5:
            raise ValueError("level must be in [0, 1.5]")
        for n in (list(STEM_LANES) if stem == "all" else [stem]):
            c.lanes[STEM_LANES[n]].set(at_bar * BAR, level, ramp_bars * BAR)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if "stem_" in e]}


STEM_PRESETS = {
    "instrumental": {"vocals": 0.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
    "acapella": {"vocals": 1.0, "drums": 0.0, "bass": 0.0, "other": 0.0},
    "drumless": {"vocals": 1.0, "drums": 0.0, "bass": 1.0, "other": 1.0},
    "bassless": {"vocals": 1.0, "drums": 1.0, "bass": 0.0, "other": 1.0},
    "full": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
}


@tool()
def stem_preset(clip: str, preset: Literal["instrumental", "acapella", "drumless", "bassless", "full"], at_bar: Pos = 0.0,
                ramp_bars: Len = 0.0) -> dict:
    """Set all four stem faders at once from `at_bar`: instrumental (no vocals), acapella (vocals only), drumless, bassless or full.
    Turns the clip's stems on (needs separate_stems first). ramp_bars fades between the previous state and the preset."""
    with _lock:
        c = clip_of(clip)
        at_bar, ramp_bars = at_of(c, at_bar), len_of(ramp_bars, "ramp_bars")
        if not c.stems:
            _need_stems(info(c.track), True)
            c.stems = True
        for n, level in STEM_PRESETS[preset].items():
            c.lanes[STEM_LANES[n]].set(at_bar * BAR, level, ramp_bars * BAR)
        save()
    return {"clip": clip, "preset": preset, "events": [e for e in describe_events(c) if "stem_" in e]}


@tool()
def crossfade(out_clip: str, in_clip: str, at_bar: Pos, bars: Len, curve: Literal["equal_power", "linear", "slow_in", "cut"] = "equal_power") -> dict:
    """Move the faders between two decks. at_bar is a bar of out_clip; the incoming clip must overlap it in time (see apply_transition
    for a full blend with EQ). Does not touch EQ, so combine with set_eq to avoid bass clashes."""
    with _lock:
        o, i = clip_of(out_clip), clip_of(in_clip)
        at_bar, bars = at_of(o, at_bar), len_of(bars)
        fn_in, fn_out = {
            "equal_power": (lambda p: np.sin(p * np.pi / 2), lambda p: np.cos(p * np.pi / 2)),
            "linear": (lambda p: p, lambda p: 1 - p),
            "slow_in": (lambda p: p ** 2, lambda p: np.cos(p * np.pi / 2)),
            "cut": (lambda p: float(p >= 0.5), lambda p: float(p < 0.5)),
        }[curve]
        b0 = at_bar * BAR
        ib0 = conv_beat(o, b0, i)
        ib1 = conv_beat(o, b0 + bars * BAR, i)
        if not i.lanes["volume"].points:
            i.lanes["volume"].hold(0.0)  # silent until the fade starts
        o.lanes["volume"].curve(b0, bars * BAR, fn_out, 4)
        i.lanes["volume"].curve(ib0, ib1 - ib0, fn_in, 4)
        save()
    return {"out": out_clip, "in": in_clip, "incoming_beat_range": [round(ib0, 2), round(ib1, 2)]}


@tool()
def list_fx() -> dict:
    """Available effects, what the beats multiplier controls for each, and their extra params with defaults."""
    return {"beats_values": [BEAT_LABELS[b] for b in BEAT_DIVISIONS],
            "fx": {n: {"mode": s.mode, "description": s.description, "beats_controls": s.uses_beats or "(not tempo-synced)", "params": s.params}
                   for n, s in fxm.CATALOG.items()}}


@tool()
def add_fx(clip: str, fx: str, at_bar: Pos, bars: Len, beats: Beats = 1.0, wet: float = 1.0, params: dict | None = None,
           stem: Literal["vocals", "drums", "bass", "other"] | None = None) -> dict:
    """Switch a tempo-synced effect on for `bars` bars from `at_bar` (see list_fx). beats in {1/16 ... 32} sets the sync division;
    wet 0-1 is the FX dry/wet. Tails of echo / reverb ring out after the effect ends. stem: apply the effect to one stem only
    (e.g. echo the vocals, keep drums dry); the clip must play its stems (stems=True after separate_stems)."""
    if fx not in fxm.CATALOG:
        raise ValueError(f"unknown fx {fx!r}; one of {list(fxm.CATALOG)}")
    with _lock:
        c = clip_of(clip)
        at_bar, bars = at_of(c, at_bar), len_of(bars)
        if stem and not c.stems:
            raise OpenMixingError("stems_off", f"clip {clip} plays the full mix, so a {stem}-only effect has nothing to act on", "stem",
                                "call separate_stems, then edit_clip(stems=True)")
        e = FxEvent(new_id("fx"), fx, at_bar * BAR, bars * BAR, parse_beats_arg(beats), float(np.clip(wet, 0, 1)), params or {}, stem)
        c.fx.append(e)
        save()
    return {"clip": clip, "fx_id": e.id}


@tool()
def remove_fx(clip: str, fx_id: str) -> dict:
    """Remove an fx event."""
    with _lock:
        c = clip_of(clip)
        c.fx = [e for e in c.fx if e.id != fx_id]
        save()
    return {"clip": clip, "remaining_fx": [e.id for e in c.fx]}


@tool()
def add_loop(clip: str, at_bar: Pos, size_beats: Beats, bars: Len | None = None, repeats: int | None = None, roll: bool = False) -> dict:
    """Engage a loop on the deck at `at_bar` (output bar; may be fractional, e.g. 31.75 for the last beat of bar 31).

    size_beats in {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32}: the loop repeats the next size_beats of the track. Stay in the loop for
    `bars` bars or `repeats` repetitions. roll=True is a loop roll: the track keeps running underneath and playback resumes where
    it would have been (a stutter); otherwise playback continues from where the loop ended. Loops may not overlap."""
    size = parse_beats_arg(size_beats)
    if (bars is None) == (repeats is None):
        raise ValueError("give exactly one of bars / repeats")
    with _lock:
        c = clip_of(clip)
        at_bar = at_of(c, at_bar)
        length = len_of(bars) * BAR if bars is not None else size * repeats
        lp = LoopEvent(new_id("lp"), at_bar * BAR, size, length, roll)
        c.loops.append(lp)
        try:
            c.validate_loops()
        except ValueError:
            c.loops.remove(lp)
            raise
        save()
    return {"clip": clip, "loop_id": lp.id, "ends_at_bar": (lp.beat + length) / BAR}


@tool()
def remove_loop(clip: str, loop_id: str) -> dict:
    """Release/remove a loop."""
    with _lock:
        c = clip_of(clip)
        c.loops = [l for l in c.loops if l.id != loop_id]
        save()
    return {"clip": clip, "remaining_loops": [l.id for l in c.loops]}


@tool()
def clear_automation(clip: str, lane: Literal["eq_low", "eq_mid", "eq_high", "filter", "volume", "stem_vocals", "stem_drums", "stem_bass", "stem_other", "tempo", "all"] = "all",
                     from_bar: Pos = 0.0, to_bar: Pos | None = None) -> dict:
    """Reset a lane (or all lanes, fx and loops with lane="all") to neutral between two bars (default: the whole clip)."""
    with _lock:
        c = clip_of(clip)
        from_bar = at_of(c, from_bar, "from_bar")
        to_bar = None if to_bar is None else at_of(c, to_bar, "to_bar")
        a, b = from_bar * BAR, (c.length_beats if to_bar is None else to_bar * BAR)
        whole = from_bar == 0 and to_bar is None
        names = list(LANES) if lane == "all" else [lane]
        for n in names:
            if n == "tempo":
                c.tempo.points = [] if whole else [p for p in c.tempo.points if not a <= p[0] <= b]
            elif whole:
                c.lanes[n].points = []
            else:
                c.lanes[n].clear(a, b)
        if lane == "all":
            c.fx = [e for e in c.fx if not (a <= e.beat <= b)]
            c.loops = [l for l in c.loops if not (a <= l.beat <= b)]
            if whole:
                c.tempo.points = []
        save()
    return {"clip": clip, "events": describe_events(c)}


@tool()
def apply_transition(out_clip: str, in_clip: str, style: Literal["bass_swap", "long_blend", "filter_fade", "echo_out", "drop_swap", "loop_roll_out", "stem_blend"] = "bass_swap",
                     bars: Len = 16.0, at_bar: Pos | None = None, tempo_after: Literal["keep", "native"] = "keep",
                     trim_out: bool = True) -> dict:
    """Mix in_clip over out_clip like a DJ: places and tempo-syncs the incoming deck on a bar of the outgoing one and writes the EQ /
    fader / filter / fx / loop moves. Styles: bass_swap (classic), long_blend (32+ bars, harmonic), filter_fade, echo_out, drop_swap,
    loop_roll_out, stem_blend (both tracks need separate_stems: vocals out, harmonics cross, drums+bass swap, vocals in). at_bar = bar of out_clip where in_clip's bar 0 lands (default: `bars` before out_clip ends, so the blend runs to the
    end). Overwrites the transition's automation; trim_out ends the outgoing clip when the blend finishes. Read `audit_set` afterwards."""
    with _lock:
        o, i = clip_of(out_clip), clip_of(in_clip)
        at_bar = None if at_bar is None else at_of(o, at_bar)
        r = planner.apply_template(cur(), info, o, i, style, at_bar, len_of(bars), tempo_after, trim_out)
        save()
    r["templates"] = planner.TEMPLATES[style]
    return jsonable(r)


@tool()
def find_windows(track: str, side: Literal["exit", "entry"], limit: int = 5) -> dict:
    """Usable regions to leave (side="exit") or enter (side="entry") a track: outros, intros, breakdowns, quiet low-bass sections.
    Each window has start/end bar, label, energy, bass, vocal presence (needs stems), phrase alignment, score and confidence."""
    f = candidates.track_facts(lib(), track)
    ve = f["vocal_entry"]
    return jsonable({"track": f["e"]["file"], "side": side, "phrase_bars": f["phrase"], "n_bars": f["n_bars"],
                     "tempo_confidence": f["tempo_conf"], "key_confidence": f["key_conf"],
                     "vocal": {"available": bool(ve), "confidence": ve["confidence"] if ve else 0.0,
                               "confidence_components": ve["confidence_components"] if ve else None},
                     "windows": candidates.find_windows(f, side)[:limit], "versions": versions()})


@tool()
def pair_report(a: str, b: str, exit_window: str | None = None, entry_window: str | None = None) -> dict:
    """Measurements for leaving track `a` into track `b`: tempo ratio and % stretch (half/double aware), key relation and semitones
    to match, loudness difference, low-end and vocal overlap risk, energy and phrase alignment. Windows default to the best ones
    (ids from find_windows). Every block carries a confidence; nothing is judged."""
    fa, fb = candidates.track_facts(lib(), a), candidates.track_facts(lib(), b)
    pick = lambda f, side, wid: next((w for w in candidates.find_windows(f, side) if wid is None or w["id"] == wid), None)
    wa, wb = pick(fa, "exit", exit_window), pick(fb, "entry", entry_window)
    for w, wid in ((wa, exit_window), (wb, entry_window)):
        if w is None:
            raise ValueError(f"no window {wid!r}; call find_windows")
    return jsonable({**candidates.pair_report(fa, fb, wa, wb), "versions": versions()})


@tool()
def suggest_transitions(a: str, b: str, goals: list[str] | dict[str, float] | None = None, limit: int = 5, include_plan: bool = True) -> dict:
    """Several feasible transitions from `a` into `b`, ranked for the goals (smooth, energetic, dramatic, experimental; default smooth;
    a dict weights them). Each candidate has sub-scores (tempo, harmony, energy_flow, bass_clash, vocal_clash, phrase, impact,
    novelty; 1 = good), a confidence, and a plan-schema document you can pass to validate_plan / apply_plan or edit first.
    Infeasible techniques are listed with the reason (e.g. stems missing). include_plan=False returns just the scores."""
    r = candidates.suggest(lib(), a, b, goals, limit)
    r["versions"] = versions()
    if not include_plan:
        for c in r["candidates"]:
            c.pop("plan")
    return jsonable(r)


@tool()
def audit_set() -> dict:
    """Lint the set like a DJ coach: clips running past their track, off-phrase entries, bass clashes (both low bands open), key clashes,
    big tempo stretches, unstable grids, silent gaps. Fix every 'error' and 'warn' before rendering."""
    return jsonable(planner.audit(cur(), lib()))


# ============================================================== declarative plans
@tool()
def get_plan_schema() -> dict:
    """The versioned JSON schema of a set plan (units, ranges, curve types, effect names). Write a plan, `validate_plan` it, then
    `apply_plan`. Plans are data only: every operation is allowlisted and range-checked."""
    return {"version": planlib.PLAN_VERSION, "schema": planlib.plan_schema()}


@tool()
def validate_plan(plan: dict) -> dict:
    """Check a plan without touching the current set. Returns every problem at once: {code, field (JSON path), message, fix}."""
    errors = planlib.validate_plan(plan, lib())
    return {"version": planlib.PLAN_VERSION, "ok": not errors, "n_errors": len(errors), "errors": errors}


@tool()
def apply_plan(plan: dict, dry_run: bool = False) -> dict:
    """Replace the current set with the plan (atomic: if the plan has any error, nothing changes and all errors are returned).
    dry_run=True only validates and reports the resulting timeline."""
    model, errors = planlib.build_plan(plan, lib())
    if errors:
        return {"version": planlib.PLAN_VERSION, "applied": False, "n_errors": len(errors), "errors": errors}
    with _lock:
        if not dry_run:
            _state["set"] = model
            save()
        saved = _state["set"]
        _state["set"] = model
        try:
            rows = [clip_row(c) for c in model.clips]
        finally:
            _state["set"] = saved
    return jsonable({"version": planlib.PLAN_VERSION, "applied": not dry_run, "set": model.name, "clips": rows})


@tool()
def export_plan() -> dict:
    """The current set as a plan document (round-trips through apply_plan)."""
    return jsonable(planlib.export_plan(cur()))


@mcp.resource("openmixing://plan-schema")
def plan_schema_resource() -> str:
    """The set plan JSON schema."""
    return json.dumps(planlib.plan_schema(), indent=1)


# ============================================================== render & play
def _qa(meta: dict) -> dict:
    notes = []
    if meta.get("true_peak_dbtp") is not None and meta["true_peak_dbtp"] > -0.5:
        notes.append("true peak above -0.5 dBTP: lower master_gain_db")
    if meta.get("integrated_lufs") is not None and meta["integrated_lufs"] < -20:
        notes.append("quiet render; raise master_gain_db or check clip volumes")
    pk = np.array(meta.get("peaks") or [0])
    return {"notes": notes, "max_block_peak": float(pk.max())}


def _do_render(name: str, start_sec, end_sec, fmt, master_gain_db, progress=None, model: SetModel | None = None) -> dict:
    m = model or cur()
    path = lib().render_path(name, fmt)
    meta = renderer.render_set(m, info, path, start_sec, end_sec, fmt, master_gain_db, progress=progress)
    meta["set"] = m.name
    meta["qa"] = _qa(meta)
    meta["versions"] = versions()
    meta["validation"] = validate.validate_render(path, m, info, [r for r in meta["clips"]], meta["start_sec"], meta)
    meta.pop("low_db", None)
    meta["metrics"] = validate.summarize(meta["validation"], meta)
    side = {"render": path.name, "set": m.name, "metrics": meta["metrics"], "validation": meta["validation"], "window": [meta["start_sec"], meta["end_sec"]]}
    path.with_suffix(".validation.json").write_text(json.dumps(jsonable(side)))
    return meta


def _publish(meta: dict) -> None:
    """Write the sidecar the player UI reads."""
    from .ui.server import build_state
    side = Path(meta["path"]).with_suffix(".json")
    side.write_text(json.dumps(jsonable(build_state(lib(), cur(), meta))))


@tool()
async def render(start_sec: float | None = None, end_sec: float | None = None, format: Literal["flac", "wav", "mp3"] = "flac",
                 master_gain_db: float = 0.0, ctx: Context = None) -> dict:
    """Render the current set (or a time window of it) to .openmixing/renders/<set>.<fmt> with a master limiter, and report integrated
    loudness / loudness range / true peak. The full render is what open_player plays."""
    loop = asyncio.get_running_loop()
    def progress(i, n, msg):
        if ctx:
            asyncio.run_coroutine_threadsafe(ctx.report_progress(i, n, msg), loop)
    name = cur().name if start_sec is None and end_sec is None else f"{cur().name}.window"
    meta = await asyncio.to_thread(_do_render, name, start_sec, end_sec, format, master_gain_db, progress)
    if start_sec is None and end_sec is None:
        _publish(meta)
    meta.pop("peaks", None)
    return jsonable(meta)


@tool()
async def preview_transition(out_clip: str, in_clip: str, bars_before: float = 8.0, bars_after: float = 8.0, format: Literal["flac", "wav"] = "flac") -> dict:
    """Render just the overlap of two clips plus some bars of lead-in / lead-out to .openmixing/renders/ for quick checking."""
    o, i = clip_of(out_clip), clip_of(in_clip)
    (so, eo), (si, ei) = span(o), span(i)
    bar = 4 * 60.0 / info(i.track)["bpm"]
    a, b = max(si - bars_before * bar, 0.0), min(eo, ei) + bars_after * bar
    meta = await asyncio.to_thread(_do_render, f"preview_{out_clip}_{in_clip}", a, b, format, 0.0)
    meta.pop("peaks", None)
    meta["window"] = [round(a, 2), round(b, 2)]
    return jsonable(meta)


def _window(m: SetModel) -> tuple[float, float]:
    """Where two clips meet: their overlap, or the join when they do not overlap."""
    rows = renderer.timeline(m, info)
    lo, hi = max(r["start_sec"] for r in rows), min(r["end_sec"] for r in rows)
    return (lo, hi) if hi > lo else (min(r["end_sec"] for r in rows),) * 2


@tool()
async def render_variations(a: str, b: str, count: int = 3, goals: list[str] | dict[str, float] | None = None,
                            plans: list[dict] | None = None, bars_before: float = 8.0, bars_after: float = 8.0,
                            format: Literal["flac", "wav"] = "flac") -> dict:
    """Render several distinct takes of one transition, each to .openmixing/renders/<id>.<fmt> with its validation, so you can compare them.

    Takes come from the best `count` suggest_transitions candidates for a -> b with different styles (goals as there), or from your own `plans` (list of plan
    documents, e.g. edited candidates; a and b are then only used for naming). Each take renders the overlap plus bars_before / bars_after.
    The current set is not touched. Returns one compact row per take: id, style, score, window, loudness, true peak, blocking failures,
    beat drift, loudness jump, low-end overlap seconds and intent pass / fail. Use compare_renders(id1, id2) for the full diff."""
    ta, tb = lib().resolve(a), lib().resolve(b)
    takes = []
    if plans:
        takes = [{"label": f"plan{i + 1}", "plan": p, "style": p.get("name", "plan"), "bars": None, "total": None} for i, p in enumerate(plans[:count])]
    else:
        sug = await asyncio.to_thread(candidates.suggest, lib(), a, b, goals, max(count * 4, 8))
        picked: list[dict] = []
        for c in sug["candidates"]:  # distinct styles first, then distinct (style, length), so the takes actually differ
            if c["style"] not in {x["style"] for x in picked}:
                picked.append(c)
        for c in sug["candidates"]:
            if c not in picked and (c["style"], c["bars"]) not in {(x["style"], x["bars"]) for x in picked}:
                picked.append(c)
        takes = [{"label": c["id"], "plan": c["plan"], "style": c["style"], "bars": c["bars"], "total": c["total"]} for c in picked[:count]]
    rows = []
    for i, t in enumerate(takes):
        model, errs = planlib.build_plan(t["plan"], lib())
        if errs:
            rows.append({"take": t["label"], "style": t["style"], "error": errs[:3]})
            continue
        lo, hi = _window(model)
        bar = 240.0 / info(model.clips[-1].track)["bpm"]
        w0, w1 = max(lo - bars_before * bar, 0.0), hi + bars_after * bar
        rid = f"var_{ta['id'][:6]}_{tb['id'][:6]}_{i + 1}_{t['style']}" + (f"_{t['bars']}b" if t["bars"] else "")
        meta = await asyncio.to_thread(_do_render, rid, w0, w1, format, 0.0, None, model)
        rows.append({"id": rid, "take": t["label"], "style": t["style"], "bars": t["bars"], "score": t["total"], "window_sec": [round(w0, 1), round(w1, 1)],
                     "path": meta["path"], **meta["metrics"]})
    return jsonable({"a": ta["file"], "b": tb["file"], "takes": rows, "versions": versions(),
                     "note": "metrics are measurements; judge them against what you intended. compare_renders(id1, id2) diffs two takes"})


def _side(ref: str) -> dict:
    name = Path(ref).name
    for ext in (".flac", ".wav", ".mp3", ".validation.json"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    p = lib().output_path(lib().state / "renders" / f"{lib().safe_name(name)}.validation.json", "ref")
    if not p.exists():
        have = sorted(x.name[: -len(".validation.json")] for x in p.parent.glob("*.validation.json"))
        raise OpenMixingError("not_found", f"no validated render {ref!r}", "ref", f"render it first; available: {have[:20]}")
    return json.loads(p.read_text())


@tool()
def compare_renders(a: str, b: str, detail: bool = False) -> dict:
    """Diff the validation metrics of two renders (ids from render_variations / render, or file names): loudness, true peak, clipping,
    unplanned silence, beat drift, loudness jump, low-end overlap, intent pass / fail. `delta` = b - a; `lower` says which render has
    the smaller value of a lower-is-better metric, it is not a verdict. detail=True adds each render's intent report."""
    sa, sb = _side(a), _side(b)
    out = {"a": sa["render"], "b": sb["render"], "diff": validate.compare(sa["metrics"], sb["metrics"]), "versions": versions()}
    if detail:
        out["intent_report"] = {"a": sa["validation"].get("intent_report", []), "b": sb["validation"].get("intent_report", [])}
    return jsonable(out)


@tool()
def open_player(port: int = 8765, open_browser: bool = True) -> dict:
    """Start the local web player for the rendered set (play / seek, timeline, live deck panel with EQ / fx / loop state) and return its URL.
    Render the set first."""
    from .ui.server import start
    with _lock:
        if _state["ui"] is None:
            _state["ui"] = start(lib(), port)
        url = _state["ui"]["url"]
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    rendered = any(lib().render_path(cur().name, f).exists() for f in ("flac", "wav"))
    return {"url": url, "set": cur().name, "rendered": rendered, "hint": None if rendered else "call render first"}


@tool()
def project_info() -> dict:
    """Where OpenMixing is working: folder, state dir, current set, track count, dependencies status."""
    import importlib.util
    return {"folder": str(lib().root), "state_dir": str(lib().state), "current_set": cur().name, "tracks": len(lib().scan()),
            "optional": {k: bool(importlib.util.find_spec(k)) for k in ("beat_this", "essentia", "madmom", "torch", "demucs")},
            "online_keys": {k: bool(os.environ.get(k)) for k in ("LASTFM_API_KEY", "DISCOGS_TOKEN", "ACOUSTID_API_KEY")}}


@tool()
def capabilities() -> dict:
    """What this installation supports: file formats, stems, limits, effects, transition styles, curves, position formats, optional
    dependencies and all schema / analysis versions. Call it before planning if unsure whether a feature is available."""
    import importlib.util
    import shutil
    from .analysis.analyze import AUDIO_EXT
    from .analysis.meta import AUDIO_EXTS
    return jsonable({
        "versions": versions(),
        "input_formats": sorted(e.lstrip(".") for e in AUDIO_EXT),
        "formats": {e.lstrip("."): {"name": AUDIO_EXTS[e], "tags_writable": e in taglib.WRITABLE_EXT} for e in sorted(AUDIO_EXTS)},
        "tag_fields": {"file_and_sidecar": list(taglib.FILE_FIELDS), "sidecar_only": list(taglib.SIDECAR_ONLY), "colors": list(taglib.COLORS)},
        "output_formats": ["flac", "wav", "mp3"], "sample_rate": SR,
        "stems": {"available": stemlib.available(), "model": stemlib.MODEL, "names": list(stemlib.STEMS)},
        "max_lanes": None, "max_lanes_note": "unbounded; limited by memory",
        "fx": {n: {"mode": f.mode, "uses_beats": f.uses_beats, "params": f.params} for n, f in fxm.CATALOG.items()},
        "beat_divisions": list(BEAT_LABELS.values()), "max_fx_tail_sec": fxm.MAX_TAIL_S,
        "transition_styles": list(planner.TEMPLATES), "curves": list(planlib.CURVES), "plan_ranges": planlib.RANGES,
        "mixer_lanes": {k: {"neutral": v[0], "min": v[1], "max": v[2], "unit": v[3]} for k, v in LANES.items()},
        "position_formats": ["16", "'16 bars'", "'64 beats'", "'95.5s'", "'cue:<name>'", "'section:<label>+N'"],
        "file_rules": {"read": str(lib().root), "write": str(lib().state)},
        "optional": {k: bool(importlib.util.find_spec(k)) for k in ("beat_this", "essentia", "madmom", "torch", "demucs", "pyacoustid")},
        "ffmpeg": bool(shutil.which("ffmpeg")),
    })


@mcp.resource("openmixing://guide")
def guide_resource() -> str:
    """The DJ playbook."""
    return GUIDE


@mcp.prompt()
def dj_set(goal: str = "a smooth 45 minute house set") -> str:
    """Start a DJ session."""
    return f"{GUIDE}\n\nTask: build {goal} from the tracks in this folder. Analyze, order, mix with apply_transition, audit_set, render, open_player."


def main() -> None:
    mcp.run(transport="stdio")
