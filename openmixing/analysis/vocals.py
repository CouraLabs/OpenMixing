from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from ..core.jsonio import read_json, write_json
from .analyze import ASR
from .confidence import combine, unit
from .stems import CACHE_VERSION

VERSION = 1
ENTER = 0.25
EXIT = 0.15
MERGE_GAP = 2
PROXY_CAP = 0.4


def vocal_activity(y: np.ndarray, sr: int, first_downbeat: float, bpm: float, n_bars: int) -> list[float]:
    bar = 240.0 / bpm
    rms = []
    for k in range(n_bars):
        a, b = int((first_downbeat + k * bar) * sr), int((first_downbeat + (k + 1) * bar) * sr)
        seg = y[a:b]
        rms.append(float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0)
    db = 20 * np.log10(np.maximum(np.array(rms), 1e-9) / max(max(rms, default=0), 1e-9))
    return [round(float(v), 3) for v in np.clip((db + 35) / 35, 0, 1)]


def regions(curve: list[float]) -> list[dict]:
    runs, start = [], None
    for k, v in enumerate(curve):
        if start is None and v >= ENTER:
            start = k
        elif start is not None and v < EXIT:
            runs.append([start, k])
            start = None
    if start is not None:
        runs.append([start, len(curve)])
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [{"start_bar": a, "end_bar": b, "mean": round(float(np.mean(curve[a:b])), 3)} for a, b in merged]


def _block(source: str | None, curve: list[float], conf: float, comps: dict) -> dict:
    reg = regions(curve)
    covered = sum(r["end_bar"] - r["start_bar"] for r in reg)
    return {"source": source, "curve": curve, "regions": reg, "density": round(covered / max(len(curve), 1), 3),
            "confidence": conf, "confidence_components": comps}


def proxy(S: np.ndarray, H: np.ndarray, fd: float, bar_sec: float, n_bars: int) -> dict:
    if n_bars < 1:
        return {"ratio": []}
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    mid = (freqs >= 300) & (freqs < 3500)
    r = H[mid].sum(0) / np.maximum(S.sum(0), 1e-9)
    fps = ASR / 512
    edges = [int((fd + k * bar_sec) * fps) for k in range(n_bars + 1)]
    bars = [r[a:max(b, a + 1)].mean() for a, b in zip(edges, edges[1:])]
    return {"ratio": [round(float(v), 4) for v in bars]}


def _proxy_block(ratio: list[float]) -> dict:
    v = np.array(ratio)
    lo, hi = np.percentile(v, 10), np.percentile(v, 90)
    curve = [round(float(x), 3) for x in np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)]
    conf, comps = combine({"contrast": unit(float(hi - lo), 0, 0.3), "proxy_cap": PROXY_CAP})
    return _block("proxy", curve, min(conf, PROXY_CAP), comps)


def vocal_entry(lib, e: dict, n_bars: int) -> dict | None:
    if not e.get("stems"):
        return None
    f = lib.state / "analysis" / f"{e['id']}.vocal.json"
    key = [e["bpm"], e["first_downbeat"], n_bars, CACHE_VERSION, VERSION]
    hit = read_json(f, {})
    if hit.get("key") == key and "confidence" in hit:
        return hit
    y, sr = sf.read(e["stems"]["vocals"], dtype="float32", always_2d=True)
    y = y.mean(1)
    curve = vocal_activity(y, sr, e["first_downbeat"], e["bpm"], n_bars)
    peak_db = 20 * np.log10(max(float(np.sqrt((y ** 2).mean())), 1e-9))
    c = np.array(curve) if curve else np.zeros(1)
    conf, comps = combine({"stem_level": unit(peak_db, -50, -20), "contrast": float(np.percentile(c, 90) - np.percentile(c, 10))})
    hit = {"key": key, **_block("stem", curve, conf, comps)}
    write_json(f, hit)
    return hit


def proxy_block(d: dict) -> dict:
    ratio = d.get("vocal_proxy", {}).get("ratio")
    return _proxy_block(ratio) if ratio else _block(None, [], 0.0, {})


def vocal_block(lib, d: dict, e: dict) -> dict:
    if (hit := vocal_entry(lib, e, e["n_bars"])):
        return {k: v for k, v in hit.items() if k != "key"}
    return proxy_block(d)
