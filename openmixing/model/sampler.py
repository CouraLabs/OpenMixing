from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.errors import OpenMixingError

KINDS = ("oneshot", "loop")
MAX_SECONDS = {"oneshot": 30.0, "loop": 120.0}
MAX_VOICES = 16


@dataclass
class SampleSlot:
    path: str
    kind: str = "oneshot"
    root_bpm: float | None = None
    gain_db: float = 0.0
    pitch_semitones: float = 0.0
    choke_group: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_seconds(kind: str, seconds: float, field: str = "path") -> None:
    if seconds > MAX_SECONDS[kind]:
        raise OpenMixingError("sample_too_long", f"a {kind} may last {MAX_SECONDS[kind]:g} s, this one lasts {seconds:.1f} s", field,
                              "trim it with start_sec / end_sec")


def max_voices(spans: list[tuple[float, float]]) -> int:
    edges = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans], key=lambda e: (e[0], e[1]))
    n = peak = 0
    for _, d in edges:
        n += d
        peak = max(peak, n)
    return peak


def check_voices(spans: list[tuple[float, float]], field: str = "triggers") -> None:
    if (n := max_voices(spans)) > MAX_VOICES:
        raise OpenMixingError("polyphony_exceeded", f"{n} sample voices sound at once, the limit is {MAX_VOICES}", field,
                              "shorten the triggers or remove some of them")
