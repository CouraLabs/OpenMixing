from __future__ import annotations

from ..core.units import BAR

KINDS = {"intro": "intro", "build": "up", "drop": "chorus", "groove": "chorus", "breakdown": "down", "outro": "outro"}


def phrase_beats(structure: dict, src_start_beat: float, length_beats: float) -> list[dict]:
    step = structure.get("phrase_bars") or 8
    out = []
    for s in structure["sections"]:
        edges = list(range(s["start_bar"], s["end_bar"], step)) + [s["end_bar"]]
        if len(edges) > 2 and edges[-1] - edges[-2] <= step / 2:
            edges.pop(-2)
        for a, b in zip(edges, edges[1:]):
            lo, hi = max(a * BAR - src_start_beat, 0.0), min(b * BAR - src_start_beat, length_beats)
            if hi > lo:
                out.append({"kind": KINDS.get(s["label"], "chorus"), "beat": lo, "end_beat": hi})
    return out
