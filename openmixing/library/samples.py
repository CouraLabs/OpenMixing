from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from ..analysis.analyze import ASR, beat_grid
from ..core.decode import decode
from ..core.errors import OpenMixingError
from ..core.units import SR
from ..model.sampler import check_seconds

PEAK = 0.891


class SampleStore:
    def sample_path(self, path: Path | str, field: str = "path") -> Path:
        for base in (self.root, self.state.resolve() / "samples"):
            try:
                return self._allowed(path, base, field, "samples", "the tracks folder")
            except OpenMixingError:
                pass
        raise OpenMixingError("path_not_allowed", f"{str(path)!r} is outside the tracks folder and .openmixing/samples/", field,
                              f"use a file name or a path relative to {self.root}")

    def sample_seconds(self, path: Path | str, field: str = "path") -> float:
        p = self.sample_path(path, field)
        try:
            return float(sf.info(p).duration)
        except Exception as e:
            raise OpenMixingError("unreadable_sample", f"{p.name} cannot be read as audio: {e}", field, "use a wav / flac / mp3 / ogg file") from e

    def import_sample(self, path: Path | str, kind: str, root_bpm: float | None, start_sec: float, end_sec: float | None) -> dict:
        src = self.sample_path(path)
        if not src.is_file():
            raise OpenMixingError("file_not_found", f"no sample file {src}", "path", "check the name with list_tracks or the folder listing")
        try:
            audio = decode(str(src))
        except Exception as e:
            raise OpenMixingError("unreadable_sample", f"{src.name} cannot be decoded: {e}", "path", "use a wav / flac / mp3 / ogg file") from e
        audio = audio[:, int(start_sec * SR) : audio.shape[1] if end_sec is None else int(end_sec * SR)]
        if audio.shape[1] < 2:
            raise OpenMixingError("empty_sample", "nothing left after trimming", "start_sec", "check start_sec / end_sec")
        check_seconds(kind, audio.shape[1] / SR)
        peak = float(np.abs(audio).max())
        if peak < 1e-6:
            raise OpenMixingError("silent_sample", f"{src.name} is silent", "path", "pick another file")
        audio = (audio * np.float32(PEAK / peak)).astype(np.float32)
        if kind == "loop" and root_bpm is None:
            root_bpm = self.loop_bpm(audio)
        dest = self.state / "samples" / f"{hashlib.blake2b(audio.tobytes(), digest_size=8).hexdigest()}.wav"
        if not dest.exists():
            sf.write(dest, audio.T, SR, subtype="FLOAT")
        return {"path": str(dest), "seconds": audio.shape[1] / SR, "root_bpm": root_bpm}

    @staticmethod
    def loop_bpm(audio: np.ndarray) -> float:
        sec = audio.shape[1] / SR
        bpm = beat_grid(soxr.resample(audio.mean(0), SR, ASR).astype(np.float32))["bpm"]
        return max(round(sec * bpm / 60), 1) * 60.0 / sec
