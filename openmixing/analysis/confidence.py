from __future__ import annotations

import numpy as np


def unit(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0))


def combine(components: dict[str, float | None], mode: str = "mean") -> tuple[float, dict[str, float]]:
    comps = {k: round(float(v), 3) for k, v in components.items() if v is not None}
    vals = list(comps.values())
    if not vals:
        return 0.0, comps
    return round(float(np.prod(vals) if mode == "product" else np.mean(vals)), 3), comps


def attach(block: dict, components: dict[str, float | None], mode: str = "mean") -> dict:
    block["confidence"], block["confidence_components"] = combine(components, mode)
    return block
