from __future__ import annotations

from typing import Literal

from ...analysis.key import harmonic_neighbours
from ...analysis.meta import UNSUPPORTED_AUDIO_EXTS, format_report
from ...core.errors import OpenMixingError, classify
from ...library.store import select
from ..app import tool
from ..state import LOCK, lib
from ..views import jsonable


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
def set_cue(track: str, name: str, position: float | str, color: str | None = None, note: str = "") -> dict:
    """Create or move a named hot cue on a track (persisted). `position` is any position format: bars, "64 beats", "95.5s",
    "section:drop+4". Cues can then be used anywhere a position is accepted as "cue:<name>"."""
    t = lib().resolve(track)
    bar, n = lib().to_bars(t["id"], position, "position"), lib().effective(t["id"])["n_bars"]
    if not 0 <= bar <= n:
        raise OpenMixingError("out_of_range", f"cue {bar:g} is outside the track (0 to {n:g} bars)", "position", f"use a bar between 0 and {n:g}")
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
def find_tracks(query: str | None = None, bpm_min: float | None = None, bpm_max: float | None = None, camelot: str | None = None,
                compatible_with_camelot: str | None = None, energy_min: int | None = None, energy_max: int | None = None,
                genre: str | None = None, crate: str | None = None, rating_min: int | None = None, color: str | None = None, tag: str | None = None,
                sort_by: Literal["bpm", "energy", "key", "name", "duration"] = "name") -> dict:
    """Filter analyzed tracks. `rating_min` / `color` / `tag` match the tags set with set_tags. `compatible_with_camelot` (e.g. "8A") keeps same
    / adjacent / relative keys. `genre` matches
    tags and cached online genres. `crate` keeps only the tracks of that smart crate."""
    ok_keys = None
    in_crate = {r["id"] for r in select(lib(), lib().store.crate(crate)["rules"])} if crate else None
    if compatible_with_camelot:
        n = harmonic_neighbours(compatible_with_camelot)
        ok_keys = {k for v in n.values() for k in v}
    rows = []
    for t in lib().scan():
        if not lib().cached(t) or (in_crate is not None and t["id"] not in in_crate):
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
        if (rating_min and ut.get("rating", 0) < rating_min) or (color and ut.get("color") != color.lower()):
            continue
        if tag and tag.strip().lower() not in ut.get("tags", []):
            continue
        rows.append({"id": t["id"], "file": t["file"], "bpm": round(e["bpm"], 2), "camelot": e["camelot"], "energy": e["energy"],
                     "duration_sec": e["duration"],
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
    with LOCK:
        return jsonable(lib().set_user_tags(track, dict(title=title, artist=artist, album=album, genre=genre, year=year, comment=comment,
                                                       label=label, rating=rating, color=color, isrc=isrc, tags=tags, notes=notes),
                                            write_to_file, backup, clear))


@tool()
def check_files() -> dict:
    """Format report for every audio file in the tracks folder: format, codec, sample rate, bit depth, channels, duration, whether it is
    lossless, whether tags can be written into it, and warnings (resampling, downmix, low bitrate). Corrupt or unsupported files are listed
    with an error code and a fix instead of failing the whole call. Files with audio extensions OpenMixing cannot read are listed as ignored."""
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
