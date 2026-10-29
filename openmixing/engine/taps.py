from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from ..core.units import SR
from ..dsp import eq as eqm
from ..model.clip import Clip
from ..model.tempo import TempoMap

BLOCK = SR // 2
CHROMA_HP_HZ = 150.0
HP = butter(4, CHROMA_HP_HZ, "highpass", fs=SR, output="sos")


def block_db(seg: np.ndarray, offset: int, n_blocks: int, blk: int, prev: np.ndarray | None = None) -> np.ndarray:
    out = np.full(n_blocks, -120.0, np.float32) if prev is None else prev
    for k in range(0, seg.shape[1], blk):
        j = (offset + k) // blk
        if j >= n_blocks:
            break
        out[j] = max(out[j], 10 * np.log10(float((seg[:, k : k + blk] ** 2).mean()) + 1e-12))
    return out


def low_curve(seg: np.ndarray, offset: int, n_blocks: int, blk: int, prev: np.ndarray | None) -> np.ndarray:
    return block_db(eqm.split_bands(seg)[0], offset, n_blocks, blk, prev)


def chroma_blocks(seg: np.ndarray, offset: int, n_blocks: int) -> np.ndarray:
    out = np.zeros((12, n_blocks), np.float16)
    lead = offset % BLOCK
    y = sosfilt(HP, np.pad(seg.mean(0), (lead, 0))).astype(np.float32)
    if len(y) < BLOCK:
        y = np.pad(y, (0, BLOCK - len(y)))
    c = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=BLOCK, hop_length=BLOCK, center=False, norm=None)
    c = c / np.maximum(c.max(axis=0, keepdims=True), 1e-9)
    j0 = offset // BLOCK
    n = max(min(c.shape[1], n_blocks - j0), 0)
    out[:, j0 : j0 + n] = c[:, :n]
    return out


def estimated_offset(curve: list[float], clip: Clip, tm: TempoMap, first: int, count: int) -> np.ndarray:
    starts = (first + np.arange(count) * BLOCK) / SR
    bars = np.clip(((clip.src_start_beat + tm.sec_to_beat(starts)) // 4).astype(int), 0, len(curve) - 1)
    return 10 * np.log10(np.maximum(np.asarray(curve)[bars], 1e-3))


class Meters:
    def __init__(self, n_blocks: int):
        self.n = n_blocks
        self.vocal_db: dict[str, np.ndarray] = {}
        self.vocal_method: dict[str, str] = {}
        self.vocal_conf: dict[str, float] = {}
        self.chroma: dict[str, np.ndarray] = {}
        self.chroma_db: dict[str, np.ndarray] = {}

    def add(self, clip: Clip, info: dict, tm: TempoMap, seg: np.ndarray, tap: dict, lo: int, g0: int, offset: int, na: int) -> None:
        cut = lambda a: a[:, lo - g0 :][:, : seg.shape[1]]
        tonal = cut(tap["tonal"]) if "tonal" in tap else seg
        self.chroma[clip.id] = chroma_blocks(tonal, offset, self.n)
        self.chroma_db[clip.id] = block_db(sosfilt(HP, tonal), offset, self.n, BLOCK)
        if "vocals" in tap:
            self.vocal_db[clip.id] = block_db(cut(tap["vocals"]), offset, self.n, BLOCK)
            self.vocal_method[clip.id], self.vocal_conf[clip.id] = "measured", 1.0
        elif info.get("vocal_curve"):
            db = block_db(seg, offset, self.n, BLOCK)
            j0 = offset // BLOCK
            first = na + (lo - g0) - offset % BLOCK
            cnt = min(self.n - j0, int(np.ceil((seg.shape[1] + offset % BLOCK) / BLOCK)))
            db[j0 : j0 + cnt] += estimated_offset(info["vocal_curve"], clip, tm, first, cnt)
            self.vocal_db[clip.id] = db
            self.vocal_method[clip.id], self.vocal_conf[clip.id] = "estimated", float(info.get("vocal_conf", 0.0))

    def export(self) -> dict:
        return {"vocal_db": self.vocal_db, "vocal_method": self.vocal_method, "vocal_conf": self.vocal_conf, "chroma": self.chroma,
                "chroma_db": self.chroma_db, "meter_block_sec": BLOCK / SR}
