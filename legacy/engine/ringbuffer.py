"""Single-producer / single-consumer frame ring buffer. The reader never blocks or allocates."""
from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, frames: int, channels: int = 2):
        self.buf = np.zeros((frames, channels), dtype=np.float32)
        self.cap = frames
        self.w = 0  # total frames written (only the feeder updates)
        self.r = 0  # total frames read (only the callback updates)

    @property
    def available(self) -> int:
        return self.w - self.r

    @property
    def space(self) -> int:
        return self.cap - self.available

    def write(self, x: np.ndarray) -> int:
        """x: (n, channels). Writes as much as fits; returns frames written."""
        n = min(len(x), self.space)
        i = self.w % self.cap
        first = min(n, self.cap - i)
        self.buf[i : i + first] = x[:first]
        self.buf[: n - first] = x[first:n]
        self.w += n
        return n

    def read_into(self, out: np.ndarray) -> int:
        """Fill `out` (n, channels); returns frames actually available (rest is zeroed)."""
        n = min(len(out), self.available)
        i = self.r % self.cap
        first = min(n, self.cap - i)
        out[:first] = self.buf[i : i + first]
        out[first:n] = self.buf[: n - first]
        out[n:] = 0
        self.r += n
        return n
