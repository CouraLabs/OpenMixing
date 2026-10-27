"""Tempo-synced DJ effects modelled on Serato DJ Pro's DJ-FX.

Every effect that has a time parameter takes `beats` in {1/16 ... 32}: delay time (echo family), decay
(reverb), LFO period (flanger, phaser, chorus, lfo_filter, tremolo), gate period (transform), slice length
(repeater, reverse) or stop time (brake). Effects are either
  send   - the dry signal keeps playing and the effect output is added (echo, ping_pong, delay, reverb ...)
  insert - the effect replaces the dry signal, cross-faded by `wet` (flanger, phaser, gate, brake ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pedalboard import Bitcrush, Chorus, Distortion, LadderFilter, Phaser, Reverb
from scipy.signal import lfilter

from ..models import BEAT_DIVISIONS, SR

MAX_TAIL_S = 12.0
EDGE = int(0.005 * SR)


@dataclass
class FxSpec:
    name: str
    mode: str          # "send" | "insert"
    description: str
    uses_beats: str    # what the beats multiplier controls ("" = nothing)
    params: dict       # extra params with defaults


CATALOG: dict[str, FxSpec] = {s.name: s for s in [
    FxSpec("echo", "send", "Damped feedback delay (Serato Echo)", "delay time", {"feedback": 0.5, "damping_hz": 6000}),
    FxSpec("echo_out", "send", "Echo with the dry signal cut: the deck 'echoes out' (Serato Echo Out)", "delay time",
           {"feedback": 0.65, "damping_hz": 5000}),
    FxSpec("delay", "send", "Clean feedback delay (Serato Delay)", "delay time", {"feedback": 0.35}),
    FxSpec("ping_pong", "send", "Stereo ping-pong delay (Serato Ping Pong Delay)", "delay time",
           {"feedback": 0.55, "damping_hz": 7000}),
    FxSpec("reverb", "send", "Room reverb (Serato Reverb)", "decay time", {"damping": 0.5}),
    FxSpec("epic_reverb", "send", "Huge, long reverb (Serato Epic Reverb)", "decay time", {"damping": 0.3}),
    FxSpec("flanger", "insert", "Jet flanger (Serato Flanger)", "LFO period", {"depth": 0.7, "feedback": 0.6}),
    FxSpec("phaser", "insert", "Phaser (Serato Phaser)", "LFO period", {"depth": 0.8, "feedback": 0.4}),
    FxSpec("chorus", "insert", "Chorus (Serato Chorus)", "LFO period", {"depth": 0.3}),
    FxSpec("lfo_filter", "insert", "Low-pass filter swept by a tempo-synced LFO (Serato LFO Filter)", "LFO period",
           {"min_hz": 250, "max_hz": 12000, "resonance": 0.4}),
    FxSpec("tremolo", "insert", "Tempo-synced volume LFO (Serato Tremolo)", "LFO period", {"depth": 1.0}),
    FxSpec("transform", "insert", "Rhythmic gate (Serato Transform)", "gate period", {"duty": 0.5}),
    FxSpec("repeater", "insert", "Beat repeater / stutter: captures the first slice and repeats it (Serato Repeater)",
           "slice length", {}),
    FxSpec("reverse", "insert", "Plays the region backwards in slices", "slice length", {}),
    FxSpec("brake", "insert", "Turntable stop (Serato Braker); silent once stopped", "stop time", {"curve": 1.0}),
    FxSpec("bitcrush", "insert", "Bit/sample-rate reduction (Serato Bitcrush)", "", {"bits": 6, "downsample": 4}),
    FxSpec("distortion", "insert", "Overdrive (Serato Distortion)", "", {"drive_db": 20}),
]}


@dataclass
class FxOut:
    y: np.ndarray
    dry_cut: bool = False


def _tail_for(beat_sec: float, beats: float, feedback: float | None = None) -> int:
    t = beats * beat_sec
    if feedback is not None:
        reps = np.log(1e-3) / np.log(max(min(feedback, 0.95), 0.05))
        t *= reps
    return int(min(t, MAX_TAIL_S) * SR)


def _feedback_delay(x: np.ndarray, delay_n: int, feedback: float, damping_hz: float | None,
                    pingpong: bool = False) -> np.ndarray:
    """Repeats only (no dry). Block recursion: repeat k = feedback * damp(input k-1 + repeat k-1)."""
    n = x.shape[1]
    D = max(int(delay_n), 1)
    src = x.mean(0, keepdims=True) if pingpong else x
    out = np.zeros_like(x)
    if damping_hz:
        a = float(np.exp(-2 * np.pi * damping_hz / SR))
    zi = np.zeros((src.shape[0], 1))
    prev = None
    for k in range(-(-n // D)):
        i, j = k * D, min(k * D + D, n)
        if prev is None:
            wet = np.zeros((src.shape[0], j - i), np.float32)
        else:
            wet = prev[:, : j - i]
            if damping_hz:
                wet, zi = lfilter([1 - a], [1, -a], wet, axis=-1, zi=zi)
            wet = (wet * feedback).astype(np.float32)
            if pingpong:
                out[k % 2, i:j] = wet[0]
            else:
                out[:, i:j] = wet
        prev = np.zeros((src.shape[0], D), np.float32)
        prev[:, : j - i] = src[:, i:j] + wet
    return out


def _echo(region, beat_sec, beats, p, tail, name):
    D = int(beats * beat_sec * SR)
    fb = float(p["feedback"])
    tail = _tail_for(beat_sec, beats, fb) if tail is None else tail
    x = np.pad(region, ((0, 0), (0, tail)))
    wet = _feedback_delay(x, D, fb, p.get("damping_hz"), pingpong=(name == "ping_pong"))
    return FxOut(wet / max(fb, 1e-3), dry_cut=(name == "echo_out"))  # first repeat at unity


def _reverb(region, beat_sec, beats, p, name):
    decay = beats * beat_sec
    room = float(np.clip(decay / 8.0, 0.2, 1.0)) if name == "reverb" else 0.97
    tail = int(min(max(decay, 1.0) * (1.5 if name == "reverb" else 2.5), MAX_TAIL_S) * SR)
    x = np.pad(region, ((0, 0), (0, tail)))
    rv = Reverb(room_size=room, damping=float(p["damping"]), wet_level=1.0, dry_level=0.0, width=1.0)
    return FxOut(rv(x, SR))


def _rate(beat_sec, beats, lo=0.05, hi=20.0):
    return float(np.clip(1.0 / (beats * beat_sec), lo, hi))


def _lfo_filter(region, beat_sec, beats, p):
    out = np.empty_like(region)
    f = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=float(p["max_hz"]), resonance=float(p["resonance"]))
    lo, hi = float(p["min_hz"]), float(p["max_hz"])
    period = beats * beat_sec * SR
    B = 128
    for i in range(0, region.shape[1], B):
        ph = 0.5 - 0.5 * np.cos(2 * np.pi * (i + B / 2) / period)
        f.cutoff_hz = float(hi * (lo / hi) ** ph)
        out[:, i : i + B] = f(region[:, i : i + B], SR, reset=False)
    return FxOut(out)


def _brake(region, beat_sec, beats, p):
    L = region.shape[1]
    dur = int(min(beats * beat_sec * SR, L))
    t = np.arange(dur, dtype=np.float64)
    c = float(p["curve"])
    u = t / dur
    pos = dur * (1 - (1 - u) ** (1 + c)) / (1 + c)  # integral of (1-u)^c: speed 1 -> 0
    out = np.zeros_like(region)
    for ch in range(region.shape[0]):
        out[ch, :dur] = np.interp(pos, np.arange(L), region[ch])
    out[:, :dur] *= np.linspace(1, 0.0, dur, dtype=np.float32) ** 0.3
    return FxOut(out)


def _slices(region, beat_sec, beats, repeat: bool):
    L = region.shape[1]
    n = max(int(beats * beat_sec * SR), 16)
    out = np.empty_like(region)
    f = min(int(0.002 * SR), n // 4)
    fade = np.ones(n, np.float32)
    if f:
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
    if repeat:
        base = region[:, :n] if region.shape[1] >= n else np.pad(region, ((0, 0), (0, n - L)))
        base = base * fade
        reps = -(-L // n)
        return np.tile(base, (1, reps))[:, :L]
    for i in range(0, L, n):
        seg = region[:, i : i + n][:, ::-1]
        out[:, i : i + seg.shape[1]] = seg * fade[: seg.shape[1]] if seg.shape[1] == n else seg
    return out


def render(name: str, region: np.ndarray, beat_sec: float, beats: float, params: dict) -> FxOut:
    """Process `region` (2, L). Returned y is (2, L) or (2, L + tail) for effects with a tail."""
    spec = CATALOG[name]
    p = {**spec.params, **{k: v for k, v in (params or {}).items() if k in spec.params}}
    L = region.shape[1]
    t = np.arange(L) / SR
    if name in ("echo", "echo_out", "delay", "ping_pong"):
        return _echo(region, beat_sec, beats, p, None, name)
    if name in ("reverb", "epic_reverb"):
        return _reverb(region, beat_sec, beats, p, name)
    if name == "flanger":
        fx = Chorus(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_delay_ms=2.0,
                    feedback=float(p["feedback"]), mix=0.6)
        return FxOut(fx(region, SR))
    if name == "phaser":
        fx = Phaser(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_frequency_hz=1300.0,
                    feedback=float(p["feedback"]), mix=0.7)
        return FxOut(fx(region, SR))
    if name == "chorus":
        fx = Chorus(rate_hz=_rate(beat_sec, beats), depth=float(p["depth"]), centre_delay_ms=7.0, feedback=0.0, mix=0.5)
        return FxOut(fx(region, SR))
    if name == "lfo_filter":
        return _lfo_filter(region, beat_sec, beats, p)
    if name == "tremolo":
        g = 1 - float(p["depth"]) * (0.5 - 0.5 * np.cos(2 * np.pi * t / (beats * beat_sec)))
        return FxOut(region * g.astype(np.float32))
    if name == "transform":
        ph = (t / (beats * beat_sec)) % 1.0
        gate = (ph < float(p["duty"])).astype(np.float32)
        k = max(int(0.002 * SR), 1)
        gate = np.convolve(gate, np.ones(k, np.float32) / k, mode="same")
        return FxOut(region * gate)
    if name == "repeater":
        return FxOut(_slices(region, beat_sec, beats, repeat=True))
    if name == "reverse":
        return FxOut(_slices(region, beat_sec, beats, repeat=False))
    if name == "brake":
        return _brake(region, beat_sec, beats, p)
    if name == "bitcrush":
        levels = 2 ** (int(p["bits"]) - 1)
        y = np.round(region * levels) / levels
        k = max(int(p["downsample"]), 1)
        if k > 1:
            y = np.repeat(y[:, ::k], k, axis=1)[:, :L]
        return FxOut(y.astype(np.float32))
    if name == "distortion":
        return FxOut(Distortion(drive_db=float(p["drive_db"]))(region, SR))
    raise KeyError(name)


def apply(x: np.ndarray, name: str, s0: int, s1: int, beat_sec: float, beats: float, wet: float,
          params: dict | None = None) -> np.ndarray:
    """Apply an effect over x[:, s0:s1] of a deck buffer that has trailing room for the tail. In place."""
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
    w = np.float32(wet) * ramp
    if spec.mode == "send":
        x[:, s0:s1] = (x[:, s0:s1] * (1 - w if o.dry_cut else 1.0)) + y[:, :L] * w
        tl = min(y.shape[1] - L, x.shape[1] - s1)
        if tl > 0:
            fade = np.ones(tl, np.float32)
            f = min(EDGE * 4, tl)
            fade[-f:] = np.linspace(1, 0, f)
            x[:, s1 : s1 + tl] += y[:, L : L + tl] * np.float32(wet) * fade
    else:
        x[:, s0:s1] = region * (1 - w) + y[:, :L] * w
    return x
