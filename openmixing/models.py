"""Shared constants."""
from __future__ import annotations

SR = 48_000  # sample rate everywhere

# The tempo-sync multipliers (in beats) usable for fx and loops, as in Serato DJ Pro's beat multiplier.
BEAT_DIVISIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
BEAT_LABELS = {1 / 16: "1/16", 1 / 8: "1/8", 1 / 4: "1/4", 1 / 2: "1/2", 1.0: "1", 2.0: "2", 4.0: "4", 8.0: "8",
               16.0: "16", 32.0: "32"}


def parse_beats(value) -> float:
    """Accept 0.25, '1/4', '1/16', 8 ... and validate against BEAT_DIVISIONS."""
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            n, d = s.split("/", 1)
            v = float(n) / float(d)
        else:
            v = float(s)
    else:
        v = float(value)
    for b in BEAT_DIVISIONS:
        if abs(b - v) < 1e-9:
            return b
    raise ValueError(f"beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {value!r}")
