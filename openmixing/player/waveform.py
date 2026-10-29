from __future__ import annotations

import numpy as np

from ..core.decode import decode
from ..core.units import SR
from ..engine.segments import segments
from ..model.tempo import TempoMap

WF_BLOCK_SEC, WF_HOP_SEC = 0.02, 0.01


def _bands(frames, n: int, total: int) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(n, 1 / SR)
    masks = {"low": freqs < 200, "mid": (freqs >= 200) & (freqs < 2500), "high": freqs >= 2500}
    win = np.hanning(n).astype(np.float32)
    out = {k: np.empty(total) for k in masks}
    for i, f in frames:
        p = np.abs(np.fft.rfft(f * win, axis=1)) ** 2
        for k, m in masks.items():
            out[k][i:i + len(f)] = np.sqrt(p[:, m].sum(axis=1))
    return {k: v / max(np.percentile(v, 95), 1e-9) for k, v in out.items()}


def band_envelope(path: str, block_sec: float) -> dict[str, list[float]]:
    x = decode(path).mean(axis=0)
    n = int(round(block_sec * SR))
    blocks = -(-len(x) // n)
    x = np.pad(x, (0, blocks * n - len(x))).reshape(blocks, n)
    out = _bands(((i, x[i:i + 256]) for i in range(0, blocks, 256)), n, blocks)
    return {k: [round(float(v), 3) for v in a] for k, a in out.items()}


def track_features(path: str) -> dict[str, np.ndarray]:
    x = decode(path).mean(axis=0)
    hop, n = int(WF_HOP_SEC * SR), int(8 * WF_HOP_SEC * SR)
    nf = len(x) // hop
    peak = np.abs(x[:nf * hop]).reshape(nf, hop).max(axis=1)
    xp = np.pad(x, (n // 2, n))
    frames = ((i, xp[(np.arange(i, min(i + 512, nf)) * hop)[:, None] + np.arange(n)]) for i in range(0, nf, 512))
    out = _bands(frames, n, nf)
    return {"a": np.minimum(peak * 100, 120), **{k[0]: v * 100 for k, v in out.items()}}


def clip_waveform(clip, e: dict, tm: TempoMap, feats: dict[str, np.ndarray]) -> dict:
    src_bpm, fd = float(e["bpm"]), float(e.get("first_downbeat", 0.0))
    segs = segments(clip)
    n = int(np.ceil(float(tm.beat_to_sec(clip.length_beats)) / WF_BLOCK_SEC))
    beats = np.minimum(tm.sec_to_beat((np.arange(n) + 0.5) * WF_BLOCK_SEC), clip.length_beats)
    src = np.empty(n)
    for i, b in enumerate(beats):
        g = next((g for g in segs if g[1] <= b <= g[2]), segs[-1])
        src[i] = g[3] + ((b - g[1]) % g[4] if g[0] == "loop" else b - g[1])
    fr = np.round((fd + src * 60.0 / src_bpm) / WF_HOP_SEC).astype(int)
    ok = (fr >= 0) & (fr < len(feats["a"]))
    fr = np.clip(fr, 0, len(feats["a"]) - 1)
    return {"b": WF_BLOCK_SEC, **{k: np.where(ok, v[fr], 0).round().astype(int).tolist() for k, v in feats.items()}}
