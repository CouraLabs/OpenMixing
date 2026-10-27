"""Offline transition renderer (plan.md §2.2): Rubber Band tempo ramp + pedalboard fx."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pedalboard import Delay, HighpassFilter, LowpassFilter, Pedalboard, time_stretch

from ..models import SR, Track, TransitionPlan
from ..planner.validator import tempo_ratio, validate_plan

TAIL_SECONDS = 4.0
BLOCK = 256


class PlanError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


@dataclass
class TransitionRender:
    a_start: int          # sample in A where the window (and pre-loop) begins
    pre_loop: np.ndarray  # played between A[:a_start] and the window
    window: np.ndarray    # (2, T) window audio
    tail: np.ndarray      # echo tails, to be mixed over the start of B's resumed audio
    b_resume: int         # sample in B where unprocessed playback resumes


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad with silence / trim so x has exactly n samples."""
    if x.shape[1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def _stretch(x: np.ndarray, factor, semitones: float = 0.0) -> np.ndarray:
    return time_stretch(np.ascontiguousarray(x, dtype=np.float32), SR, factor,
                        pitch_shift_in_semitones=semitones, high_quality=True)


def tempo_curve_incoming(plan: TransitionPlan, b: Track, bar_n: float, T: int) -> np.ndarray:
    """Per-output-sample playback speed of the incoming deck: hold at target, then ramp to native (1.0)."""
    r0 = tempo_ratio(plan.target_bpm, b.bpm) if plan.beatmatch else 1.0
    hold = plan.tempo_ramp.get("hold_bars", 0) * bar_n
    ramp = plan.tempo_ramp.get("ramp_bars", 0) * bar_n
    return np.interp(np.arange(T), [0, hold, hold + ramp], [r0, r0, 1.0])


CONTEXT = 4096  # source samples of context around each stretch chunk
XFADE = 480     # half-length (samples) of the crossfade between chunks


def _stretch_variable(track: Track, start: int, speed_out: np.ndarray, semitones: float,
                      chunk: int) -> tuple[np.ndarray, int]:
    """Play `track` from `start` so output sample k advances the source by ~speed_out[k].

    pedalboard's time_stretch is only trustworthy with a scalar ratio: array ratios shift the audio by
    thousands of samples, and even scalar ratios drift ~0.1% (the total length is exact, the beats
    inside it are not), which accumulates to tens of ms over a 32-bar hold. So the window is stretched
    in half-bar chunks, each at a constant (mean) ratio with context on both sides, placed at its exact
    output position and crossfaded. That bounds the beat error to ~1 ms and turns the tempo ramp into a
    fine staircase. Returns (audio of exactly len(speed_out) samples, source samples consumed).
    """
    T = len(speed_out)
    speed = np.empty(T)
    for o0 in range(0, T, chunk):
        speed[o0 : o0 + chunk] = speed_out[o0 : o0 + chunk].mean()
    pos = np.concatenate([[0.0], np.cumsum(speed)])  # source offset at the start of each output sample
    out = np.zeros((track.audio.shape[0], T), dtype=np.float32)
    for o0 in range(0, T, chunk):
        o1 = min(o0 + chunk, T)
        e0, e1 = max(o0 - XFADE, 0), min(o1 + XFADE, T)
        s0, s1 = int(round(pos[e0])), int(round(pos[e1]))
        src = track.slice(start + s0 - CONTEXT, start + s1 + CONTEXT)
        ratio = float(speed[o0])
        if abs(ratio - 1) < 1e-9 and semitones == 0:
            piece = src[:, CONTEXT : CONTEXT + e1 - e0]
        else:
            y = _stretch(src, ratio, semitones)
            piece = y[:, int(round(CONTEXT / ratio)) :]
        piece = fit(piece, e1 - e0)
        w = np.ones(e1 - e0, dtype=np.float32)
        n_in, n_out = (min(2 * XFADE, len(w)) for _ in range(2))
        if e0 > 0:
            w[:n_in] = np.linspace(0, 1, n_in, endpoint=False)
        if e1 < T:
            w[len(w) - n_out :] = np.linspace(1, 0, n_out, endpoint=False)
        out[:, e0:e1] += piece * w
    return out, int(round(pos[-1]))


def _ramped(points: list[tuple[float, float]], T: int) -> np.ndarray:
    xs, ys = zip(*points)
    return np.interp(np.arange(T), xs, ys).astype(np.float32)


def _bass_curve(events, deck: str, bar_n: float, T: int) -> np.ndarray:
    ramp = bar_n / 8
    pts = [(0.0, 1.0)]
    for ev in sorted((e for e in events if e["deck"] == deck), key=lambda e: e["bar"]):
        g = 0.0 if ev["bass"] == "kill" else 1.0
        x = ev["bar"] * bar_n
        pts += [(max(x, pts[-1][0]), pts[-1][1]), (max(x, pts[-1][0]) + ramp, g)]
    return _ramped(pts, T)


def _eq(x: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Bass band gain; high band is the exact complement so gain=1 is transparent."""
    low = Pedalboard([LowpassFilter(150), LowpassFilter(150)])(x, SR)
    return (x - low) + low * gain[None, :x.shape[1]]


def _sweep(x: np.ndarray, kind: str, fx: dict, bar_n: float) -> np.ndarray:
    cls = LowpassFilter if kind == "lowpass_sweep" else HighpassFilter
    default = (18000, 800) if kind == "lowpass_sweep" else (20, 2000)
    f0, f1 = fx.get("start_hz", default[0]), fx.get("end_hz", default[1])
    s, e = (b * bar_n for b in fx["bars"])
    plugin = cls(f0)
    out = np.empty_like(x)
    for i in range(0, x.shape[1], BLOCK):
        frac = np.clip((i + BLOCK / 2 - s) / (e - s), 0, 1)
        plugin.cutoff_frequency_hz = float(f0 * (f1 / f0) ** frac)
        out[:, i : i + BLOCK] = plugin(x[:, i : i + BLOCK], SR, reset=False)
    return out


def _echo_send(x: np.ndarray, fx: dict, bar_n: float, beat_n: float) -> np.ndarray:
    """Echo 'throw': feed the deck into a feedback delay during fx['bars']; wet bypasses the fader."""
    s, e = (b * bar_n for b in fx["bars"])
    ramp = 64
    send = _ramped([(0, 0), (s, 0), (s + ramp, 1), (e, 1), (e + ramp, 0)], x.shape[1])
    delay = Delay(delay_seconds=fx.get("delay_beats", 0.75) * beat_n / SR,
                  feedback=fx.get("feedback", 0.5), mix=1.0)
    return delay(x * send[None, :], SR)


def _fader(curve: str, bars, L: int, bar_n: float, T: int) -> tuple[np.ndarray, np.ndarray]:
    s, e = (0, L) if bars is None else bars
    p = np.clip((np.arange(T) - s * bar_n) / max((e - s) * bar_n, 1), 0, 1)
    if curve == "linear":
        return (1 - p).astype(np.float32), p.astype(np.float32)
    # Equal-power, lifted by 1.25 and clipped at unity: both decks stay nearly full through the middle
    # of the blend (bass is already swapped by EQ), avoiding the loudness dip of a plain crossfade.
    return (np.clip(1.25 * np.cos(p * np.pi / 2), 0, 1).astype(np.float32),
            np.clip(1.25 * np.sin(p * np.pi / 2), 0, 1).astype(np.float32))


def render_transition(a: Track, b: Track, plan: TransitionPlan) -> TransitionRender:
    errors = validate_plan(plan, a, b)
    if errors:
        raise PlanError(errors)

    L, bpb = plan.length_bars, a.beats_per_bar
    bar_n = bpb * 60.0 / plan.target_bpm * SR  # window bar length in samples
    T = int(round(L * bar_n))
    tail_n = int(TAIL_SECONDS * SR)
    a_start, b_start = a.bar_pos(plan.from_bar), b.bar_pos(plan.to_bar)

    # Outgoing deck: constant stretch to the shared tempo (1.0 in the normal case).
    ra = tempo_ratio(plan.target_bpm, a.bpm) if plan.beatmatch else 1.0
    if ra == 1:
        a_win = a.slice(a_start, a_start + T)
    else:
        a_win, _ = _stretch_variable(a, a_start, np.full(T, ra), 0.0, int(bar_n / 2))

    # Incoming deck: held at target tempo, then ramped back to native inside the window.
    speed = tempo_curve_incoming(plan, b, bar_n, T)
    b_win, n_src = _stretch_variable(b, b_start, speed, float(plan.pitch_shift_semitones), int(bar_n / 2))

    decks = {"from": a_win, "to": b_win}
    for deck in decks:
        decks[deck] = _eq(decks[deck], _bass_curve(plan.eq_automation, deck, bar_n, T))
    for fx in plan.fx:
        if fx["type"].endswith("_sweep"):
            decks[fx["deck"]] = _sweep(decks[fx["deck"]], fx["type"], fx, bar_n)

    ga, gb = _fader(plan.crossfade["curve"], plan.crossfade.get("bars"), L, bar_n, T)
    mix = np.zeros((a_win.shape[0], T + tail_n), dtype=np.float32)
    mix[:, :T] = decks["from"] * ga + decks["to"] * gb
    for fx in plan.fx:
        if fx["type"] == "echo_out":
            padded = np.pad(decks[fx["deck"]], ((0, 0), (0, tail_n)))
            mix += _echo_send(padded, fx, bar_n, bar_n / bpb)

    pre = np.zeros((a.audio.shape[0], 0), dtype=np.float32)
    if plan.pre_loop:
        n = int(round(plan.pre_loop["bars"] * a.bar_samples))
        pre = np.tile(a.slice(a_start - n, a_start), (1, plan.pre_loop["repeats"]))

    return TransitionRender(a_start, pre, mix[:, :T], mix[:, T:], b_start + n_src)
