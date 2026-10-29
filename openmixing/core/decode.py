from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soxr

from .errors import OpenMixingError
from .units import SR


def _run(cmd: list[str], path: str, **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, check=True, **kw)
    except FileNotFoundError as e:
        if Path(path).exists():
            raise OpenMixingError("missing_dependency", f"{cmd[0]} is not installed", fix="install ffmpeg (provides ffmpeg and ffprobe)") from e
        raise
    except subprocess.CalledProcessError as e:
        err = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")
        raise OpenMixingError("io_error", f"cannot decode {Path(path).name}: {err.strip() or 'ffmpeg failed'}", "path",
                              "check the file is a valid, non-empty audio file") from e


def decode(path: str) -> np.ndarray:
    probe = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path], path, text=True)
    out = probe.stdout.strip().rstrip(",")
    if not out:
        raise OpenMixingError("io_error", f"{Path(path).name} has no audio stream", "path", "use an audio file")
    src_sr = int(out)
    raw = _run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(src_sr), "-"], path).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    audio = audio[: len(audio) // 2 * 2].reshape(-1, 2)
    if src_sr != SR:
        audio = soxr.resample(audio, src_sr, SR, quality="VHQ")
    return np.ascontiguousarray(audio.T, dtype=np.float32)
