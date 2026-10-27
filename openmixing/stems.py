"""Stem separation: split a track into vocals / drums / bass / other with Demucs (htdemucs).

The stems are cached next to the analysis as 48 kHz FLAC files with exactly the sample count of the decoded track, so they
share its beat grid and a clip can swap the full mix for the sum of its stems without any re-alignment.
"""
from __future__ import annotations

import numpy as np
import soxr

from .models import SR

STEMS = ("vocals", "drums", "bass", "other")
STEM_INFO = {
    "vocals": "lead and backing vocals",
    "drums": "drums and percussion",
    "bass": "bass lines and sub (the low end lives here)",
    "other": "everything else: synths, pads, keys, guitars, samples",
}
MODEL = "htdemucs"
CACHE_VERSION = 2   # 1 = the old 3-stem layout (vocals / drums_bass / harmonic)
LEGACY_STEMS = {"drums_bass": ("drums", "bass"), "harmonic": ("other",)}   # old stem name -> new stems it covers


def available() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "demucs"))


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _resample(x: np.ndarray, a: int, b: int) -> np.ndarray:
    return x if a == b else np.ascontiguousarray(soxr.resample(x.T, a, b, quality="HQ").T, dtype=np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:, :n] if x.shape[1] >= n else np.pad(x, ((0, 0), (0, n - x.shape[1])))


def separate(audio: np.ndarray) -> dict[str, np.ndarray]:
    """audio: (2, n) float32 at SR. Returns {stem: (2, n) float32 at SR}; the stems sum back to ~the input."""
    if not available():
        raise RuntimeError("stem separation needs Demucs: pip install 'openmixing[stems]'  (installs torch + demucs)")
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(MODEL)
    model.eval()
    msr, n = int(model.samplerate), audio.shape[1]
    x = torch.from_numpy(_resample(audio, SR, msr))
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((x - mean) / std)[None]

    def run(device: str):
        return apply_model(model, x, device=device, split=True, overlap=0.25, shifts=0, progress=False)[0]

    dev = _device()
    try:
        y = run(dev)
    except Exception:
        if dev == "cpu":
            raise
        y = run("cpu")  # mps / cuda can lack ops or memory: the CPU is slower but always works
    y = (y * std + mean).cpu().numpy()
    src = {name: _fit(_resample(y[i], msr, SR), n) for i, name in enumerate(model.sources)}
    return {name: src[name] for name in STEMS}


def bass_carrier(info: dict) -> str:
    """Which stem holds the low end: 'bass', or 'drums_lowband' when the bass stem came out empty (Demucs files 808s under drums)."""
    rep = info.get("stem_quality") or {}
    q = (rep.get("stems") or {}).get("bass") or {}
    return "drums_lowband" if q.get("empty") else "bass"


def low_stem(info: dict) -> str:
    """Stem name whose level gates the low end of this track."""
    return "bass" if bass_carrier(info) == "bass" else "drums"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def quality_report(audio: np.ndarray, stems: dict[str, np.ndarray]) -> dict:
    """Estimate separation quality from the stems alone: energy share, bleed and reconstruction error per stem.

    bleed = strongest absolute correlation of the stem's 100 ms RMS envelope with another stem (shared dynamics suggest
    leakage); residual = energy of (mix - sum of stems) relative to the mix. confidence falls with bleed and residual."""
    blk = SR // 10
    n = audio.shape[1] // blk
    env = {k: np.sqrt((v.mean(0)[: n * blk].reshape(n, blk) ** 2).mean(1)) for k, v in stems.items()}
    energy = {k: float((v.astype(np.float64) ** 2).sum()) for k, v in stems.items()}
    total = sum(energy.values()) or 1.0
    resid = float(1.0 - np.clip(_rms(audio - sum(stems.values())) / max(_rms(audio), 1e-9), 0, 1))
    out = {}
    for k in stems:
        share = energy[k] / total
        others = [abs(float(np.corrcoef(env[k], env[o])[0, 1])) for o in stems if o != k and env[k].std() > 0 and env[o].std() > 0]
        bleed = max(others, default=0.0)
        empty = share < 0.01
        conf = 0.0 if empty else float(np.clip((1 - bleed) * 0.6 + resid * 0.4, 0, 1))
        out[k] = {"energy_share": round(share, 3), "bleed": round(bleed, 3), "empty": empty, "confidence": round(conf, 3),
                  "confidence_components": {"low_bleed": round(1 - bleed, 3), "reconstruction": round(resid, 3)}}
    return {"stems": out, "reconstruction_match": round(resid, 3),
            "note": "estimates from the stems only (no ground truth): treat a low confidence or an empty stem as a warning, not a verdict"}
