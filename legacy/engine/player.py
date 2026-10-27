"""Gapless playback: feeder thread -> ring buffer -> sounddevice callback (plan.md §2.2).

The callback only copies from the ring buffer. The transition is rendered in a worker process
while the outgoing track plays; the feeder blocks on the render only after A's pre-window audio
has been queued.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..models import SR, Track, TransitionPlan
from ..render.mix import segments
from ..render.transition import render_transition
from .ringbuffer import RingBuffer

BUFFER_SECONDS = 12
BLOCK_FRAMES = 4096


class Player:
    def __init__(self, device=None):
        self.ring = RingBuffer(BUFFER_SECONDS * SR)
        self.underruns = 0
        self.device = device
        self.done = threading.Event()  # feeder has queued everything
        self.error: BaseException | None = None

    def _callback(self, outdata, frames, time_info, status):
        if self.ring.read_into(outdata) < frames and not self.done.is_set():
            self.underruns += 1

    def _feed(self, blocks):
        try:
            for block in blocks:
                block = np.ascontiguousarray(block.T)
                for i in range(0, len(block), BLOCK_FRAMES):
                    chunk = block[i : i + BLOCK_FRAMES]
                    while self.ring.space < len(chunk):
                        time.sleep(0.01)
                    self.ring.write(chunk)
        except BaseException as e:  # surfaced to the main thread
            self.error = e
        finally:
            self.done.set()

    def play(self, blocks) -> None:
        """Play an iterator of (2, n) float32 arrays gaplessly; returns when drained."""
        import sounddevice as sd

        feeder = threading.Thread(target=self._feed, args=(blocks,), daemon=True)
        feeder.start()
        while self.ring.available < min(2 * SR, self.ring.cap // 2) and not self.done.is_set():
            time.sleep(0.01)
        with sd.OutputStream(samplerate=SR, channels=2, dtype="float32", latency="high",
                             blocksize=BLOCK_FRAMES, device=self.device, callback=self._callback):
            while not (self.done.is_set() and self.ring.available == 0):
                time.sleep(0.05)
            time.sleep(0.5)  # let the device drain
        if self.error:
            raise self.error


def play_set(a: Track, b: Track, plan: TransitionPlan, device=None) -> int:
    """Play A -> transition -> B. Returns the underrun count."""
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        job = pool.submit(render_transition, a, b, plan)

        def blocks():
            # Stream A up to the window while the worker renders.
            a_start = a.bar_pos(plan.from_bar)
            for i in range(0, a_start, SR):
                yield master_block(a.slice(i, min(i + SR, a_start)))
            r = job.result()  # render deadline: must be done by now
            for seg in segments(a, b, r)[1:]:
                for i in range(0, seg.shape[1], SR):
                    yield master_block(seg[:, i : i + SR])

        player = Player(device)
        player.play(blocks())
        return player.underruns


_limiter = None


def master_block(x: np.ndarray) -> np.ndarray:
    """Stateful master limiter applied block by block."""
    global _limiter
    if _limiter is None:
        from pedalboard import Limiter

        _limiter = Limiter(threshold_db=-1.0)
    return _limiter(x, SR, reset=False)


def _audio_main(a, b, plan, device, q):
    q.put(play_set(a, b, plan, device))


def play_set_in_process(a: Track, b: Track, plan: TransitionPlan, device=None) -> int:
    """Run playback in its own process so analysis/rendering in the parent never starves the callback."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_audio_main, args=(a, b, plan, device, q))
    proc.start()
    underruns = q.get()
    proc.join()
    return underruns
