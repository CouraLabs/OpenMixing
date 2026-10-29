from __future__ import annotations

from hashlib import blake2b

import numpy as np

from .hits import crash, flat_noise, impact, snare
from .riser import downlifter, riser

PAN = (0.0, -1.0, 1.0)
SWEEP = {"f0": (200.0, 20.0, 20000.0), "f1": (8000.0, 20.0, 20000.0), "tone": (0.0, 0.0, 1.0), "color": (0.0, 0.0, 2.0),
         "widen": (1.0, 0.0, 1.0), "pan": PAN}
KINDS = {
    "riser": {"fn": riser, "align": "end", "params": SWEEP, "description": "band-passed noise (+ tone) swept f0 -> f1 with a rising level"},
    "downlifter": {"fn": downlifter, "align": "start", "params": SWEEP, "description": "a riser played backwards: falling sweep and level"},
    "impact": {"fn": impact, "align": "start", "description": "sine sweep drop (boom) + noise burst, exponential decay", "params": {
        "f_start": (150.0, 20.0, 2000.0), "f_end": (40.0, 20.0, 2000.0), "decay_s": (1.5, 0.05, 10.0), "noise_mix": (0.3, 0.0, 1.0), "pan": PAN}},
    "crash": {"fn": crash, "align": "start", "description": "high-passed noise burst, exponential decay", "params": {
        "hp_hz": (3000.0, 200.0, 12000.0), "decay_s": (1.5, 0.05, 10.0), "pan": PAN}},
    "snare": {"fn": snare, "align": "start", "description": "sine body + noise burst one-shot (used by add_roll)", "params": {
        "pitch_hz": (180.0, 60.0, 600.0), "decay_s": (0.15, 0.02, 2.0), "pan": PAN}},
    "noise": {"fn": flat_noise, "align": "start", "description": "steady noise bed", "params": {"color": (0.0, 0.0, 2.0), "pan": PAN}},
}


def seed_of(trigger_id: str) -> int:
    return int.from_bytes(blake2b(trigger_id.encode(), digest_size=4).digest(), "big")


def generate(kind: str, trigger_id: str, n: int, params: dict) -> np.ndarray:
    spec = KINDS[kind]
    p = {k: v[0] for k, v in spec["params"].items()} | {k: v for k, v in params.items() if k in spec["params"]}
    y = spec["fn"](n, np.random.default_rng(seed_of(trigger_id)), **p)
    return (y / max(float(np.abs(y).max()), 1e-9)).astype(np.float32)


def describe() -> dict:
    return {k: {"default_align": s["align"], "description": s["description"],
                "params": {n: {"default": d, "min": lo, "max": hi} for n, (d, lo, hi) in s["params"].items()}} for k, s in KINDS.items()}
