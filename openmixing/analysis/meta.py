"""File properties (ffprobe) and embedded tags (mutagen)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mutagen
import mutagen.id3

from ..errors import OpenMixingError
from ..tags import WRITABLE_EXT

AUDIO_EXTS = {".mp3": "MP3", ".flac": "FLAC", ".wav": "WAV", ".aiff": "AIFF", ".aif": "AIFF", ".m4a": "AAC/M4A", ".aac": "AAC",
              ".ogg": "OGG Vorbis", ".opus": "Opus", ".alac": "ALAC"}
LOSSLESS = ("flac", "alac", "wavpack", "ape")
UNSUPPORTED_AUDIO_EXTS = {".wma", ".ape", ".wv", ".mka", ".ac3", ".dts", ".amr", ".mid", ".midi"}


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else round(f, 3)
    except (TypeError, ValueError):
        return v


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except ValueError:
        return {}
    fmt = j.get("format", {})
    audio = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "container": fmt.get("format_name"), "codec": audio.get("codec_name"), "profile": audio.get("profile"),
        "sample_rate": _num(audio.get("sample_rate")), "channels": audio.get("channels"),
        "channel_layout": audio.get("channel_layout"), "bit_depth": _num(audio.get("bits_per_raw_sample") or
                                                                             audio.get("bits_per_sample") or None) or None,
        "bitrate_kbps": round(float(fmt["bit_rate"]) / 1000) if fmt.get("bit_rate") else None,
        "duration_sec": _num(fmt.get("duration")), "size_bytes": _num(fmt.get("size")),
        "has_cover_art": any(s.get("codec_type") == "video" for s in j.get("streams", [])),
        "ffprobe_tags": {k.lower(): v for k, v in (fmt.get("tags") or {}).items()},
    }


def read_tags(path: str) -> dict:
    """Common tags plus anything DJ-relevant (BPM / key / ISRC / label / comment / Serato presence)."""
    out: dict = {}
    try:
        easy = mutagen.File(path, easy=True)
        if easy is not None and easy.tags:
            for k, v in easy.tags.items():
                out[k.lower()] = v[0] if isinstance(v, list) and len(v) == 1 else v
        raw = mutagen.File(path)
        if raw is not None and raw.tags is not None:
            keys = list(raw.tags.keys())
            if isinstance(raw.tags, mutagen.id3.ID3):  # WAV / AIFF have no "easy" tags; read the ID3 frames directly
                for fid, name in (("TIT2", "title"), ("TPE1", "artist"), ("TALB", "album"), ("TCON", "genre"), ("TDRC", "date")):
                    if fid in raw.tags and name not in out:
                        out[name] = str(raw.tags[fid])
            out["has_serato_data"] = any("serato" in str(k).lower() for k in keys)
            for k in keys:
                ks = str(k)
                if ks.startswith(("TKEY", "TBPM", "TSRC", "TPUB", "COMM", "TCOM", "TIT3")) or ks.lower() in (
                        "initialkey", "bpm", "isrc", "label", "©cmt"):
                    out.setdefault(ks.split(":")[0].lower(), str(raw.tags[k]))
    except Exception as e:  # unreadable tags never block analysis
        out["tag_error"] = str(e)
    out.setdefault("title", Path(path).stem)
    return {k: (str(v) if not isinstance(v, (str, int, float, bool, list)) else v) for k, v in out.items()}


def format_report(path: str, pr: dict | None = None) -> dict:
    """What the file is and whether OpenMixing can use it: format, codec, sample rate, bit depth, channels, duration, warnings.
    Raises OpenMixingError (unsupported_format / corrupt_file) when it cannot be used."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in AUDIO_EXTS:
        raise OpenMixingError("unsupported_format", f"{p.name}: {ext or 'no extension'} is not a supported audio format", "track",
                            f"convert to one of {sorted(AUDIO_EXTS)} (e.g. ffmpeg -i in.wma out.flac)")
    pr = pr if pr is not None else probe(str(path))
    if not pr or not pr.get("codec") or not pr.get("duration_sec"):
        raise OpenMixingError("corrupt_file", f"{p.name} could not be decoded (ffprobe found no audio stream)", "track",
                            "re-export or re-download the file, or remove it from the tracks folder")
    codec = pr["codec"]
    lossless = codec.startswith("pcm_") or codec in LOSSLESS
    warn = []
    if pr.get("sample_rate") not in (None, 48000):
        warn.append(f"{pr['sample_rate']} Hz is resampled to 48000 Hz for analysis and render")
    if (pr.get("channels") or 2) > 2:
        warn.append(f"{pr['channels']} channels are downmixed to stereo")
    if pr.get("channels") == 1:
        warn.append("mono is duplicated to stereo")
    if float(pr["duration_sec"]) < 10:
        warn.append("very short file: beat grid and key estimates will be unreliable")
    if not lossless and (pr.get("bitrate_kbps") or 999) < 128:
        warn.append(f"low bitrate ({pr['bitrate_kbps']} kbps): expect artifacts when stretching")
    return {"extension": ext, "format": AUDIO_EXTS[ext], "container": pr.get("container"), "codec": codec, "lossless": lossless,
            "sample_rate": pr.get("sample_rate"), "bit_depth": pr.get("bit_depth"), "channels": pr.get("channels"),
            "duration_sec": pr.get("duration_sec"), "bitrate_kbps": pr.get("bitrate_kbps"), "supported": True,
            "tags_writable": ext in WRITABLE_EXT, "warnings": warn}
