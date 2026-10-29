from __future__ import annotations

import numpy as np

from ..core.units import SR
from ..dsp import dynamics
from ..model.clip import Clip
from ..model.events import DuckEvent
from ..model.tempo import TempoMap

END_FADE = int(0.005 * SR)


def key_stem(ev: DuckEvent) -> str | None:
    return ev.key.split(":", 1)[1] if ev.key.startswith("self:") else None


def key_clip(ev: DuckEvent) -> str | None:
    return None if ev.key.startswith("self:") else ev.key


def duck_window(sig: np.ndarray, key: np.ndarray, ev: DuckEvent, clip: Clip, tm: TempoMap, na: int) -> None:
    s0 = max(tm.beat_to_sample(ev.beat) - na, 0)
    s1 = min(tm.beat_to_sample(min(ev.beat + ev.duration_beats, clip.length_beats)) - na, sig.shape[1])
    if s1 <= s0:
        return
    x = sig[:, s0:s1].copy()
    y = dynamics.duck(x, key[..., s0:s1], ev.depth_db, ev.attack_ms, ev.release_ms, ev.threshold_db)
    r = np.ones(s1 - s0, np.float32)
    f = min(END_FADE, len(r))
    r[-f:] = np.linspace(1, 0, f, dtype=np.float32)
    sig[:, s0:s1] = x * (1 - r) + y * r


def duck_stems(sigs: dict[str, np.ndarray], ducks, clip: Clip, tm: TempoMap, na: int) -> None:
    for ev, key in ducks:
        if ev.stem is None or ev.stem not in sigs:
            continue
        k = key if key is not None else sigs.get(key_stem(ev))
        if k is not None:
            duck_window(sigs[ev.stem], k, ev, clip, tm, na)


def duck_deck(main: np.ndarray, ducks, clip: Clip, tm: TempoMap, na: int) -> None:
    for ev, key in ducks:
        if ev.stem is None and key is not None:
            duck_window(main, key, ev, clip, tm, na)


def cross_key(ev: DuckEvent, clip: Clip, tm: TempoMap, off: int, k_len: int, render, na: int, nb: int) -> np.ndarray:
    key = np.zeros(nb - na, np.float32)
    n0 = max(na, tm.beat_to_sample(ev.beat))
    n1 = min(nb, tm.beat_to_sample(min(ev.beat + ev.duration_beats, clip.length_beats)))
    k0, k1 = max(n0 + off, 0), min(n1 + off, k_len)
    if k1 > k0:
        key[k0 - off - na : k1 - off - na] = np.abs(render((k0, k1))[:, : k1 - k0]).max(axis=0)
    return key
