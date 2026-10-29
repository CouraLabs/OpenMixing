from __future__ import annotations

import numpy as np

from ..core.units import SR
from ..dsp import eq as eqm
from ..dsp import fx as fxm
from ..dsp import stretch
from ..model.clip import Clip
from ..model.lane import STEM_LANES, Lane
from ..model.tempo import TempoMap
from .cache import AudioCache
from .ducking import duck_deck, duck_stems
from .effects import apply_fx
from .sampler import Samples
from .segments import segments
from .triggers import add_triggers
from .vinyl import play_rate

TAIL = int(fxm.MAX_TAIL_S * SR)
FADE = int(0.001 * SR)


def stem_levels(clip: Clip) -> dict[str, Lane]:
    x = np.linspace(min(0.1, clip.length_beats), clip.length_beats, max(int(clip.length_beats * 4), 2))
    return {n: clip.lanes[l] for n, l in STEM_LANES.items() if clip.lanes[l].sample(x).max() > 1e-3}


def load_audio(clip: Clip, info: dict, cache: "AudioCache"):
    if not clip.stems:
        return cache.get(info["path"])
    if not info.get("stems"):
        raise ValueError(f"clip {clip.id} plays stems but {info['file']} has " + (
            "only the old 3-stem layout: call separate_stems(force=True) to re-split into vocals / drums / bass / other"
            if info.get("stems_legacy") else "none: call separate_stems first"))
    return {n: cache.get(info["stems"][n]) for n in stem_levels(clip)}


def render_clip(clip: Clip, info: dict, audio, tm: TempoMap, win: tuple[int, int] | None = None,
                ducks=(), samples: Samples | None = None, taps: dict | None = None) -> np.ndarray:
    N = tm.beat_to_sample(clip.length_beats)
    na, nb = win or (0, N)
    na, nb = max(na, 0), min(nb, N)
    W = nb - na
    buf = np.zeros((2, W + TAIL), np.float32)
    if W <= 0:
        return buf
    gain = np.float32(10 ** ((float(info.get("gain_db", 0.0)) + clip.gain_db) / 20))
    ob = tm.sec_to_beat(np.arange(na, nb) / SR)
    main = buf[:, :W]
    stem_tails = np.zeros((2, TAIL), np.float32)
    sigs = {}
    if isinstance(audio, dict):
        for name, a in audio.items():
            lane = clip.lanes[STEM_LANES[name]]
            sigs[name] = play_source(clip, info, a, tm, na, nb) * (1.0 if lane.is_neutral() else lane.sample(ob))
        duck_stems(sigs, ducks, clip, tm, na)
        for name, sig in sigs.items():
            evs = [e for e in clip.fx if e.stem == name]
            if evs:
                sb = np.zeros((2, W + TAIL), np.float32)
                sb[:, :W] = sig
                apply_fx(sb, evs, clip, tm, na, W)
                sig, stem_tails = sb[:, :W], stem_tails + sb[:, W:]
            main += sig
    else:
        if any(e.stem for e in clip.fx):
            raise ValueError(f"clip {clip.id} has stem fx but plays the full mix: set stems=True (needs separate_stems)")
        main += play_source(clip, info, audio, tm, na, nb)
    main *= gain
    buf[:, W:] += stem_tails * gain
    lanes = clip.lanes

    def env(name):
        return None if lanes[name].is_neutral() else lanes[name].sample(ob)

    main[:] = eqm.apply_eq(main, env("eq_low"), env("eq_mid"), env("eq_high"))
    if (f := env("filter")) is not None:
        main[:] = eqm.apply_filter_knob(main, f)
    if (v := env("volume")) is not None:
        main *= v
    if taps is not None and sigs:
        scale = gain * (1.0 if v is None else v)
        taps["vocals"] = sigs.get("vocals", np.zeros_like(main)) * scale
        taps["tonal"] = sum((s for n, s in sigs.items() if n != "drums"), np.zeros_like(main)) * scale
    duck_deck(main, ducks, clip, tm, na)
    if clip.triggers:
        add_triggers(buf, main.copy(), clip, tm, na, W, samples or {})
    apply_fx(buf, [e for e in clip.fx if not e.stem], clip, tm, na, W)
    return buf


def play_source(clip: Clip, info: dict, audio: np.ndarray, tm: TempoMap, na: int, nb: int) -> np.ndarray:
    src_bpm, fd = float(info["bpm"]), float(info.get("first_downbeat", 0.0))
    buf = np.zeros((2, nb - na), np.float32)
    kl = clip.key_lock
    moving = not clip.pitch.is_neutral()
    chunk = max(int(2 * 60.0 / src_bpm * SR), 4096)

    def src_sample(beat: float) -> float:
        return (fd + beat * 60.0 / src_bpm) * SR

    prev_kind = None
    for seg in segments(clip):
        kind, o0, o1, s0 = seg[:4]
        n0, n1 = tm.beat_to_sample(o0), tm.beat_to_sample(o1)
        a, b = max(n0, na), min(n1, nb)
        if b <= a:
            prev_kind = kind
            continue
        if kind == "play":
            o_a = float(tm.sec_to_beat(a / SR))
            i0, i1 = int(o0 * tm.RES), min(int(np.ceil(o1 * tm.RES)) + 1, len(tm.bpm))
            if np.ptp(tm.bpm[i0:i1]) < 1e-6:
                speed = float(tm.bpm_at(o_a)) / src_bpm
            else:
                speed = tm.bpm_at(tm.sec_to_beat(np.arange(a, b) / SR)) / src_bpm
            semi = clip.semitones + clip.pitch.sample(tm.sec_to_beat(np.arange(a, b) / SR)) if moving else clip.semitones
            y = stretch.play(audio, src_sample(s0 + (o_a - o0)), speed, b - a, semi, kl, chunk)
            if prev_kind in ("loop", "rate"):
                f = min(FADE, y.shape[1] // 4)
                if f:
                    y[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
            buf[:, a - na : b - na] += y
        elif kind == "rate":
            buf[:, a - na : b - na] += play_rate(audio, seg[4], s0, tm, src_bpm, fd, (n0, n1), (a, b))
        else:
            size = seg[4]
            speed = float(tm.bpm_at(o0)) / src_bpm
            reps = [(tm.beat_to_sample(o0 + j * size), tm.beat_to_sample(min(o0 + (j + 1) * size, o1)))
                    for j in range(int(np.ceil((o1 - o0) / size - 1e-9)))]
            body_n = max(nj1 - nj for nj, nj1 in reps) + 2
            body = stretch.play(audio, src_sample(s0), speed, body_n, clip.semitones + clip.pitch.value_at(o0), kl, chunk)
            for nj, nj1 in reps:
                lo, hi = max(nj, na), min(nj1, nb)
                if hi <= lo:
                    continue
                piece = body[:, : nj1 - nj].copy()
                f = min(FADE, piece.shape[1] // 4)
                if f:
                    piece[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
                    piece[:, -f:] *= np.linspace(1, 0, f, dtype=np.float32)
                buf[:, lo - na : hi - na] += piece[:, lo - nj : hi - nj]
        prev_kind = kind
    return buf
