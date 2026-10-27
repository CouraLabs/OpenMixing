"""Assemble a full A -> window -> B mix; render-to-WAV mode (plan.md §2.2)."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pedalboard import Limiter, Pedalboard

from ..models import SR, Track, TransitionPlan
from .transition import TransitionRender, render_transition


def master(x: np.ndarray) -> np.ndarray:
    return Pedalboard([Limiter(threshold_db=-1.0)])(x, SR)


def segments(a: Track, b: Track, r: TransitionRender) -> list[np.ndarray]:
    """Contiguous pieces of the set in play order; the window's echo tail is mixed into B's resume."""
    post = b.slice(r.b_resume, b.n_samples)
    n = min(r.tail.shape[1], post.shape[1])
    post[:, :n] += r.tail[:, :n]
    return [a.slice(0, r.a_start), r.pre_loop, r.window, post]


def render_mix(a: Track, b: Track, plan: TransitionPlan) -> np.ndarray:
    r = render_transition(a, b, plan)
    return master(np.concatenate(segments(a, b, r), axis=1))


def render_to_wav(a: Track, b: Track, plan: TransitionPlan, path: str) -> None:
    sf.write(path, render_mix(a, b, plan).T, SR, subtype="PCM_24")


class _Writer:
    """Streams audio to disk through a stateful master limiter, carrying echo tails across segments."""

    def __init__(self, path: str, subtype: str):
        self.file = sf.SoundFile(path, "w", SR, 2, subtype=subtype)
        self.limiter = Pedalboard([Limiter(threshold_db=-1.0)])
        self.pending = np.zeros((2, 0), dtype=np.float32)
        self.written = 0

    def emit(self, x: np.ndarray) -> None:
        x = x.copy()
        n = min(x.shape[1], self.pending.shape[1])
        x[:, :n] += self.pending[:, :n]
        self.pending = self.pending[:, n:]
        for i in range(0, x.shape[1], SR * 10):
            self.file.write(np.clip(self.limiter(x[:, i : i + SR * 10], SR, reset=False), -0.98, 0.98).T)
        self.written += x.shape[1]

    def add_tail(self, tail: np.ndarray) -> None:
        m = max(tail.shape[1], self.pending.shape[1])
        out = np.zeros((2, m), dtype=np.float32)
        out[:, : tail.shape[1]] += tail
        out[:, : self.pending.shape[1]] += self.pending
        self.pending = out

    def close(self) -> None:
        if self.pending.shape[1]:
            self.emit(np.zeros((2, self.pending.shape[1]), dtype=np.float32))
        self.file.close()


def _fade(x: np.ndarray, out: bool) -> np.ndarray:
    n = min(SR, x.shape[1])
    ramp = np.linspace(1, 0, n, dtype=np.float32) if out else np.linspace(0, 1, n, dtype=np.float32)
    x = x.copy()
    if out:
        x[:, -n:] *= ramp
    else:
        x[:, :n] *= ramp
    return x


def render_set(metas: list[Track], plans: list[TransitionPlan | None], load, path: str,
               progress=None, subtype: str = "PCM_16") -> list[dict]:
    """Render a whole ordered set to `path`, holding at most two decoded tracks in memory.

    `metas` are audio-less Tracks (grid + gain); `load(meta)` returns the same Track with audio.
    A None plan means the pair can't be beatmatched: the outgoing track fades out and the next starts.
    Returns a timeline: [{index, start_s}] (output time at which each track becomes audible).
    """
    w = _Writer(path, subtype)
    timeline = [{"index": 0, "start_s": 0.0}]
    cur, pos = load(metas[0]), 0
    for i, plan in enumerate(plans):
        if progress:
            progress(i, len(plans))
        nxt = load(metas[i + 1])
        if plan is None:
            w.emit(_fade(cur.slice(pos, cur.n_samples), out=True))
            timeline.append({"index": i + 1, "start_s": w.written / SR})
            cur, pos = nxt, nxt.bar_pos(0)
            w.emit(_fade(cur.slice(pos, pos + SR), out=False))
            pos += SR
            continue
        r = render_transition(cur, nxt, plan)
        w.emit(cur.slice(pos, r.a_start))
        w.emit(r.pre_loop)
        timeline.append({"index": i + 1, "start_s": w.written / SR})
        w.emit(r.window)
        w.add_tail(r.tail)
        cur, pos = nxt, r.b_resume
    w.emit(cur.slice(pos, cur.n_samples))
    w.close()
    if progress:
        progress(len(plans), len(plans))
    return timeline
