"""Track tags: validation, the sidecar-only fields (rating, color, tags, notes) and writing common tags into the audio file."""
from __future__ import annotations

import re
from pathlib import Path

import mutagen
from mutagen import id3, mp4

from .errors import OpenMixingError

FILE_FIELDS = ("title", "artist", "album", "genre", "year", "comment", "label", "isrc")
SIDECAR_ONLY = ("rating", "color", "tags", "notes")
COLORS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "pink")
ID3_EXT = {".mp3", ".wav", ".aiff", ".aif"}
VORBIS_EXT = {".flac", ".ogg", ".opus"}
MP4_EXT = {".m4a", ".alac"}
WRITABLE_EXT = ID3_EXT | VORBIS_EXT | MP4_EXT


def clean(field: str, value):
    """Validate one tag value; returns the normalised value."""
    if field == "rating":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 5 or int(value) != value:
            raise OpenMixingError("invalid_value", f"rating must be a whole number 0-5, got {value!r}", "rating", "use 0 (none) to 5")
        return int(value)
    if field == "color":
        v = str(value).strip().lower()
        if v not in COLORS and not re.fullmatch(r"#[0-9a-f]{6}", v):
            raise OpenMixingError("invalid_value", f"unknown color {value!r}", "color", f"use one of {list(COLORS)} or #rrggbb")
        return v
    if field == "year":
        m = re.fullmatch(r"\s*(\d{4})\s*", str(value))
        if not m:
            raise OpenMixingError("invalid_value", f"year must be 4 digits, got {value!r}", "year", "e.g. 2024")
        return int(m.group(1))
    if field == "isrc":
        v = str(value).replace("-", "").strip().upper()
        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}\d{7}", v):
            raise OpenMixingError("invalid_value", f"{value!r} is not an ISRC", "isrc", "12 characters, e.g. USRC17607839")
        return v
    if field == "tags":
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) and x.strip() for x in value):
            raise OpenMixingError("invalid_value", "tags must be a list of non-empty strings", "tags", 'e.g. ["peak-time", "vocal hook"]')
        return sorted({x.strip() for x in value})
    return str(value)


def _id3_frames(f: dict) -> list:
    out = []
    simple = {"title": id3.TIT2, "artist": id3.TPE1, "album": id3.TALB, "genre": id3.TCON, "label": id3.TPUB, "isrc": id3.TSRC}
    for k, cls in simple.items():
        if k in f:
            out.append(cls(encoding=3, text=[str(f[k])]))
    if "year" in f:
        out.append(id3.TDRC(encoding=3, text=[str(f["year"])]))
    if "comment" in f:
        out.append(id3.COMM(encoding=3, lang="eng", desc="", text=[str(f["comment"])]))
    return out


def write_file(path: str, fields: dict) -> dict:
    """Write FILE_FIELDS into the audio file. Returns {"written": [...], "skipped": {field: reason}}."""
    ext = Path(path).suffix.lower()
    if ext not in WRITABLE_EXT:
        return {"written": [], "skipped": {k: f"{ext or 'this format'} cannot hold tags; kept in the sidecar only" for k in fields}}
    if ext in ID3_EXT:
        if ext == ".mp3":
            try:
                t = id3.ID3(path)
            except id3.ID3NoHeaderError:
                t = id3.ID3()
            for fr in _id3_frames(fields):
                t.setall(fr.FrameID if fr.FrameID != "COMM" else "COMM", [fr])
            t.save(path)
        else:
            f = mutagen.File(path)
            if f.tags is None:
                f.add_tags()
            for fr in _id3_frames(fields):
                f.tags.setall(fr.FrameID, [fr])
            f.save()
    elif ext in VORBIS_EXT:
        f = mutagen.File(path)
        if f is None:
            raise OpenMixingError("corrupt_file", f"{Path(path).name} could not be opened for tagging", "track", "re-encode or replace the file")
        if f.tags is None:
            f.add_tags()
        names = {"year": "date", "label": "label"}
        for k, v in fields.items():
            f.tags[names.get(k, k)] = [str(v)]
        f.save()
    else:
        f = mp4.MP4(path)
        atoms = {"title": "\xa9nam", "artist": "\xa9ART", "album": "\xa9alb", "genre": "\xa9gen", "year": "\xa9day", "comment": "\xa9cmt"}
        for k, v in fields.items():
            if k in atoms:
                f[atoms[k]] = [str(v)]
            else:
                f[f"----:com.apple.iTunes:{k.upper()}"] = [mp4.MP4FreeForm(str(v).encode())]
        f.save()
    return {"written": sorted(fields), "skipped": {}}
