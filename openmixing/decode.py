"""Decode any ffmpeg-readable file to 48 kHz stereo float32, shape (2, n)."""
from __future__ import annotations

import subprocess

import numpy as np
import soxr

from .models import SR


def decode(path: str) -> np.ndarray:
    # ffmpeg decodes at the source rate; soxr does the high-quality resample (plan.md §1).
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    src_sr = int(probe.stdout.strip().rstrip(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(src_sr), "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    if src_sr != SR:
        audio = soxr.resample(audio, src_sr, SR, quality="VHQ")
    return np.ascontiguousarray(audio.T, dtype=np.float32)
