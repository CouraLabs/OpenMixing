from __future__ import annotations

MAX_CHAIN = 6
RANGES = {"eq": (-80.0, 6.0), "filter": (-1.0, 1.0), "volume": (0.0, 1.5), "stem": (0.0, 1.5), "bpm": (40.0, 250.0),
          "gain_db": (-40.0, 12.0), "semitones": (-12.0, 12.0), "ramp_bars": (0.0, 256.0), "wet": (0.0, 1.0),
          "exponent": (0.1, 10.0), "offset_ms": (-2000.0, 2000.0), "rate_speed": (-4.0, 4.0), "rate_beats": (1 / 16, 32.0),
          "depth_db": (0.0, 40.0), "threshold_db": (-80.0, 0.0), "attack_ms": (0.1, 500.0), "release_ms": (1.0, 5000.0),
          "root_bpm": (20.0, 400.0), "sample_semitones": (-24.0, 24.0), "dissonance": (0.0, 1.0), "correlation": (-1.0, 1.0),
          "tension_level": (0.0, 1.0), "gap_beats": (0.0, 4.0)}


def num_range(key: str) -> dict:
    lo, hi = RANGES[key]
    return {"type": "number", "minimum": lo, "maximum": hi}
