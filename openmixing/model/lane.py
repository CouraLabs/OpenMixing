from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

LANES = {
    "eq_low": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "eq_mid": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "eq_high": (0.0, -80.0, 6.0, "dB (-80 = kill)"),
    "filter": (0.0, -1.0, 1.0, "-1 = full low-pass, 0 = off, +1 = full high-pass"),
    "volume": (1.0, 0.0, 1.5, "channel fader (linear)"),
    "stem_vocals": (1.0, 0.0, 1.5, "vocals stem level (linear)"),
    "stem_drums": (1.0, 0.0, 1.5, "drums stem level (linear)"),
    "stem_bass": (1.0, 0.0, 1.5, "bass stem level (linear)"),
    "stem_other": (1.0, 0.0, 1.5, "other (synths, pads, keys, guitars) stem level (linear)"),
}


STEM_LANES = {"vocals": "stem_vocals", "drums": "stem_drums", "bass": "stem_bass", "other": "stem_other"}
LEGACY_LANES = {"stem_drums_bass": ("stem_drums", "stem_bass"), "stem_harmonic": ("stem_other",)}
LEGACY_STEM_NAMES = {"drums_bass": ("drums", "bass"), "harmonic": ("other",)}
MIN_RAMP = 0.02


@dataclass
class Lane:

    neutral: float = 0.0
    points: list[list[float]] = field(default_factory=list)

    def value_at(self, beat: float) -> float:
        if not self.points:
            return self.neutral
        xs = [p[0] for p in self.points]
        i = bisect.bisect_right(xs, beat)
        if i == 0:
            return self.points[0][1]
        if i == len(xs):
            return self.points[-1][1]
        (x0, y0), (x1, y1) = self.points[i - 1], self.points[i]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (beat - x0) / (x1 - x0)

    def set(self, beat: float, value: float, ramp: float = 0.0) -> None:
        ramp = max(ramp, MIN_RAMP)
        before = self.value_at(beat)
        self.points = [p for p in self.points if p[0] < beat or p[0] > beat + ramp]
        self.points += [[beat, before], [beat + ramp, value]]
        self.points.sort(key=lambda p: p[0])

    def hold(self, value: float) -> None:
        self.points = [[0.0, value]]

    def curve(self, beat: float, length: float, fn, steps_per_beat: int = 2) -> None:
        n = max(int(round(length * steps_per_beat)), 1)
        self.points = [p for p in self.points if p[0] < beat or p[0] > beat + length]
        self.points += [[beat + length * i / n, float(fn(i / n))] for i in range(n + 1)]
        self.points.sort(key=lambda p: p[0])

    def clear(self, a: float, b: float) -> None:
        self.points = [p for p in self.points if p[0] < a or p[0] > b]
        self.points += [[a, self.neutral], [b, self.neutral]]
        self.points.sort(key=lambda p: p[0])

    def is_neutral(self) -> bool:
        return all(abs(p[1] - self.neutral) < 1e-9 for p in self.points)

    def sample(self, beats: np.ndarray) -> np.ndarray:
        if not self.points:
            return np.full(beats.shape, self.neutral, np.float32)
        xs, ys = zip(*self.points)
        return np.interp(beats, xs, ys).astype(np.float32)
