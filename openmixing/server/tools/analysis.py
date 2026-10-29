from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

from mcp.server.fastmcp import Context

from ...analysis import region as regionlib
from ...analysis.vocals import vocal_block, vocal_entry
from ...core.errors import OpenMixingError
from ...library import Library
from ..app import tool
from ..state import lib
from ..versions import versions
from ..views import jsonable


@tool()
async def analyze_track(track: str, force: bool = False, use_stems: bool = False, include_beats: bool = False, include_waveform: bool = False,
                        include_tension: bool = True, include_vocal: bool = True, max_points: int = 128,
                        region: list[float | str] | None = None, ctx: Context = None) -> dict:
    """Full analysis of one track (cached in .openmixing/). `track` is an id, file name, or unique part of the name.

    Returns: file info (codec, bitrate, ...), embedded tags, tempo (bpm, first downbeat, grid stability, drift),
    key (name, Camelot, confidence, candidates, key over time, compatible keys), loudness (LUFS, LRA, true peak,
    clipping, gain to -14 LUFS, leading/trailing silence), stereo, spectrum (band shares, centroid, MFCC, contrast,
    cutoff / transcode check), energy (1-10, danceability heuristic), structure (sections with labels, phrase length,
    per-bar energy and bass curves), mix points (cues, mix_in, mix_out), loop candidates, manual overrides, and cached
    online metadata if `lookup_online` was run. Takes ~10-30 s the first time. Every estimate carries `confidence` (0..1) and
    `confidence_components` (what it was derived from).

    use_stems=True separates the stems (Demucs, once) and re-fits the beat grid on the drums stem and the downbeat on drums + bass,
    snapping the bpm to a whole number when the fit allows: use it when the grid or bpm of a track sounds off.
    tempo carries beat_strength, grid_source ("mix" | "drums_stem") and downbeat_confidence.

    region=[start, end] (any position format, e.g. [32, "section:drop+8"]) replaces the per-bar arrays, sections and loop candidates of
    `structure` with just that range, and trims key_over_time; use get_region for more fields.

    Also returns `tension` (per-bar curve downsampled to `max_points` unless include_tension=False, weights, peaks; components via get_region),
    `vocal` (source stem / proxy / null, per-bar curve, regions, density; the proxy used without stems is capped at confidence 0.4),
    `harmony.key_by_phrase` (key per phrase; per-bar chroma via get_region what=["harmony"]) and `lead_in` (pre-drop gap) on sections.
    """
    d = await asyncio.to_thread(lib().analyze, track, force, use_stems)
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
    out["harmony"] = {k: v for k, v in d["harmony"].items() if k != "chroma_bar"}
    if include_tension:
        out["tension"] = {**d["tension"], "curve": regionlib.curve_summary(d["tension"]["curve"], max_points)}
        out["tension"].pop("components")
    else:
        out.pop("tension")
    out.pop("vocal_proxy")
    if include_vocal:
        v = vocal_block(lib(), d, eff)
        out["vocal"] = {**v, "curve": regionlib.curve_summary(v["curve"], max_points)}
    if region is not None:
        a, b = _range(d["id"], region)
        keep = {k: v for k, v in out["structure"].items() if k not in ("sections", "energy_curve", "bass_curve", "onset_curve", "loop_candidates")}
        keep["region"] = regionlib.slice_region(d, eff, a, b, ("energy", "bass", "onsets", "sections", "loops"))
        out["structure"] = keep
        r0, r1 = keep["region"]["start_sec"] - 1e-9, keep["region"]["end_sec"]
        out["key"] = {**out["key"], "key_over_time": [k for k in out["key"]["key_over_time"] if r0 <= k["start_sec"] < r1]}
        out["region"] = [a, b]
    return jsonable(out)


def _range(track: str, rng) -> tuple[float, float]:
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise OpenMixingError("invalid_value", f"region must be [start, end], got {rng!r}", "region",
                              "e.g. [32, 64] or [\"cue:drop\", \"section:breakdown\"]")
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
        vocal = vocal_entry(lib(), e, e["n_bars"])
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
def override_track(track: str, bpm: float | None = None, first_downbeat_sec: float | None = None,
                   shift_downbeat_beats: float | None = None, camelot: str | None = None, key: str | None = None,
                   gain_db: float | None = None, clear: list[str] | None = None) -> dict:
    """Correct the analysis (persisted). bpm / first_downbeat_sec fix the beat grid, shift_downbeat_beats moves the
    downbeat by whole beats (e.g. 1 when the grid is a beat late), camelot / key fix the key, gain_db sets the level
    trim. `clear` lists override names to remove (bpm, first_downbeat, camelot, key, gain_db)."""
    t = lib().resolve(track)
    lib().analyze(t["id"])
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
    return {"track": t["file"], "overrides": lib().set_override(t["id"], **kv),
            "note": "clips in existing sets use the corrected grid on the next render"}
