"""Tracklist and cue sheet of a set: where each clip starts in the rendered file, with track metadata."""
from __future__ import annotations

import json
from pathlib import Path

FPS = 75  # CD frames per second in a cue sheet


def clock(sec: float, frames: bool = False) -> str:
    sec = max(sec, 0.0)
    if frames:
        total = int(sec * FPS)
        m, rest = divmod(total, 60 * FPS)
        return f"{m:02d}:{rest // FPS:02d}:{rest % FPS:02d}"
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def rows(timeline: list[dict], info, tags, offset_sec: float | None = None) -> list[dict]:
    """One row per clip, time 0 = the start of the render (the earliest clip unless `offset_sec`)."""
    if not timeline:
        return []
    t0 = min(r["start_sec"] for r in timeline) if offset_sec is None else offset_sec
    out = []
    for r in sorted(timeline, key=lambda r: (r["start_sec"], r["clip"])):
        e, tg = info(r["track"]), tags(r["track"])
        out.append({"n": len(out) + 1, "clip": r["clip"], "time_sec": round(r["start_sec"] - t0, 3), "time": clock(r["start_sec"] - t0),
                    "end_sec": round(r["end_sec"] - t0, 3), "duration_sec": r["duration_sec"], "file": e["file"],
                    "artist": tg.get("artist"), "title": tg.get("title") or Path(e["file"]).stem, "album": tg.get("album"),
                    "label": tg.get("label"), "isrc": tg.get("isrc"), "bpm": round(e["bpm"], 2), "camelot": e["camelot"],
                    "rating": tg.get("rating"), "color": tg.get("color")})
    return out


def as_text(name: str, rs: list[dict]) -> str:
    lines = [f"{name}", ""]
    for r in rs:
        who = f"{r['artist']} - {r['title']}" if r["artist"] else r["title"]
        lines.append(f"{r['time']}  {who}  [{r['bpm']:g} BPM, {r['camelot']}]")
    return "\n".join(lines) + "\n"


def as_json(name: str, rs: list[dict], audio_file: str) -> str:
    return json.dumps({"set": name, "audio_file": audio_file, "tracks": rs}, indent=1)


def _q(s) -> str:
    return '"' + str(s or "").replace('"', "'") + '"'


def as_cue(name: str, rs: list[dict], audio_file: str) -> str:
    kind = "MP3" if audio_file.lower().endswith(".mp3") else "WAVE"
    out = [f"TITLE {_q(name)}", 'PERFORMER "OpenMixing"', f"FILE {_q(audio_file)} {kind}"]
    for r in rs:
        out += [f"  TRACK {r['n']:02d} AUDIO", f"    TITLE {_q(r['title'])}"]
        if r["artist"]:
            out.append(f"    PERFORMER {_q(r['artist'])}")
        if r["isrc"]:
            out.append(f"    ISRC {r['isrc']}")
        out.append(f"    INDEX 01 {clock(r['time_sec'], frames=True)}")
    return "\n".join(out) + "\n"
