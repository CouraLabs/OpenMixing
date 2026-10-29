from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from hashlib import blake2b
from importlib import metadata
from pathlib import Path
from typing import Callable

from ..core.jsonio import write_json
from ..engine import MP3_KBPS
from ..library import Library
from ..model.setmodel import SetModel
from ..plan import export_plan
from .fingerprint import samples, seeds, stems, tracks

LIBS = ("numpy", "scipy", "pedalboard", "librosa", "soundfile", "soxr", "demucs")


def file_hash(path: Path | str) -> str:
    h = blake2b()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record_path(render: Path) -> Path:
    return render.with_suffix(".repro.json")


def lib_versions() -> dict:
    out = {}
    for n in LIBS:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def ffmpeg_line() -> str | None:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def environment(versions: dict) -> dict:
    return {"openmixing": versions.get("openmixing"), "versions": versions, "python": platform.python_version(), "platform": platform.platform(),
            "libs": lib_versions(), "ffmpeg": ffmpeg_line()}


def build_record(model: SetModel, meta: dict, params: dict, versions: dict, lib: Library, info: Callable[[str], dict]) -> dict:
    out = Path(meta["path"])
    fmt = {"format": params["format"], **({"mp3_bitrate_kbps": MP3_KBPS} if params["format"] == "mp3" else {})}
    return {**environment(versions), "set": model.to_dict(), "plan": export_plan(model), "params": {**params, **fmt},
            "tracks": tracks(model, lib, info), "stems": stems(model, info, True), "samples": samples(model), "seeds": seeds(model),
            "output": {"file": out.name, "blake2b": file_hash(out), "duration": meta["duration_sec"], "lufs": meta["integrated_lufs"],
                       "true_peak": meta["true_peak_dbtp"]},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def write_record(path: Path, record: dict) -> None:
    write_json(record_path(path), record, indent=1)
