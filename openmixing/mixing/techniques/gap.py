from __future__ import annotations

from ...model.lane import MIN_RAMP, Lane

GAP_RAMP = 0.0625
MIN_GAP = 2 * GAP_RAMP
EXISTING_GAP_MIN = 0.25


def hold_gap(volume: Lane, resolve: float, gap: float, release: bool = True) -> None:
    if gap <= 0:
        return
    volume.set(resolve - gap, 0.0, GAP_RAMP)
    if release:
        volume.set(resolve, 1.0, MIN_RAMP)


def existing_gap(sections: list[dict], src_bar: float, tol: float = 0.5) -> float:
    for s in sections:
        if abs(s["start_bar"] - src_bar) <= tol and (g := s.get("lead_in", {}).get("gap_beats", 0.0)) >= EXISTING_GAP_MIN:
            return float(g)
    return 0.0
