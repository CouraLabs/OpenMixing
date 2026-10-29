from __future__ import annotations

import numpy as np

from . import delays, glitch, levels, modulation
from .catalog import CATALOG, EDGE, MAX_TAIL_S, FxOut, FxSpec

RENDERERS = {
    "echo": delays.echo, "echo_out": delays.echo, "delay": delays.echo, "ping_pong": delays.echo,
    "reverb": delays.reverb, "epic_reverb": delays.reverb,
    "flanger": modulation.flanger, "phaser": modulation.phaser, "chorus": modulation.chorus,
    "lfo_filter": modulation.lfo_filter, "tremolo": modulation.tremolo, "transform": modulation.transform,
    "repeater": glitch.repeater, "reverse": glitch.reverse, "brake": glitch.brake,
    "bitcrush": glitch.bitcrush, "distortion": glitch.distortion,
    "compressor": levels.compressor, "sidechain_pump": levels.sidechain_pump,
}


def render(name: str, region: np.ndarray, beat_sec: float, beats: float, params: dict) -> FxOut:
    spec = CATALOG[name]
    p = {**spec.params, **{k: v for k, v in (params or {}).items() if k in spec.params}}
    return RENDERERS[name](region, beat_sec, beats, p, name)


def apply(x: np.ndarray, name: str, s0: int, s1: int, beat_sec: float, beats: float, wet: float | np.ndarray,
          params: dict | None = None) -> np.ndarray:
    s0, s1 = max(s0, 0), min(s1, x.shape[1])
    if s1 <= s0:
        return x
    spec = CATALOG[name]
    L = s1 - s0
    region = x[:, s0:s1].copy()
    o = render(name, region, beat_sec, beats, params or {})
    y = o.y.astype(np.float32)
    ramp = np.ones(L, np.float32)
    e = min(EDGE, L // 2)
    if e:
        ramp[:e] = np.linspace(0, 1, e)
        ramp[-e:] = np.linspace(1, 0, e)
    wet = np.asarray(wet, np.float32)
    w = wet * ramp
    if spec.mode == "send":
        x[:, s0:s1] = (x[:, s0:s1] * (1 - w if o.dry_cut else 1.0)) + y[:, :L] * w
        tl = min(y.shape[1] - L, x.shape[1] - s1)
        if tl > 0:
            fade = np.ones(tl, np.float32)
            f = min(EDGE * 4, tl)
            fade[-f:] = np.linspace(1, 0, f)
            x[:, s1 : s1 + tl] += y[:, L : L + tl] * wet.reshape(-1)[-1] * fade
    else:
        x[:, s0:s1] = region * (1 - w) + y[:, :L] * w
    return x
