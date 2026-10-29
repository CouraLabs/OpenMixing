from __future__ import annotations

import math

import numpy as np

from ..core.errors import OpenMixingError

CURVES = {"energy": "energy_curve", "bass": "bass_curve", "onsets": "onset_curve"}
WHAT = ("energy", "bass", "onsets", "tension", "vocal", "harmony", "beats", "sections", "loops", "cues", "waveform")
DEFAULT_WHAT = ("energy", "bass", "sections")
MAX_BEATS = 512


class Grid:

    def __init__(self, e: dict, d: dict):
        self.fd0, self.bpm0 = float(d["tempo"]["first_downbeat_sec"]), float(d["tempo"]["bpm"])
        self.fd, self.bpm = float(e["first_downbeat"]), float(e["bpm"])

    def to_analysis(self, bar: float) -> float:
        return (self.fd + bar * 240.0 / self.bpm - self.fd0) * self.bpm0 / 240.0

    def to_effective(self, bar: float) -> float:
        return (self.fd0 + bar * 240.0 / self.bpm0 - self.fd) * self.bpm / 240.0

    def sec(self, bar: float) -> float:
        return self.fd + bar * 240.0 / self.bpm


def _b(x: float):
    x = round(float(x), 3)
    return int(x) if x == int(x) else x


def _downsample(vals: list[float], idx: list[int], max_points: int) -> tuple[list[float], list[int], int]:
    step = max(math.ceil(len(vals) / max_points), 1)
    if step == 1:
        return vals, idx, 1
    return ([round(float(np.mean(vals[i:i + step])), 3) for i in range(0, len(vals), step)], idx[::step], step)


def curve_summary(vals: list[float], max_points: int) -> dict:
    v, _, step = _downsample(vals, list(range(len(vals))), max_points)
    return {"values": v, "step_bars": step}


def slice_region(d: dict, e: dict, start: float, end: float, what=DEFAULT_WHAT, max_points: int = 128, vocal: dict | None = None,
                 cues: dict | None = None, waveform: dict | None = None) -> dict:
    g, st, t = Grid(e, d), d["structure"], d["tempo"]
    n = len(st["energy_curve"])
    a, b = g.to_analysis(start), g.to_analysis(end)
    ks = list(range(max(math.floor(a + 1e-6), 0), min(math.ceil(b - 1e-6), n)))
    out: dict = {"start_bar": _b(start), "end_bar": _b(end), "start_sec": round(g.sec(start), 3), "end_sec": round(g.sec(end), 3),
                 "grid": "override" if abs(g.bpm - g.bpm0) > 1e-9 or abs(g.fd - g.fd0) > 1e-9 else "analysis"}
    bars_of = lambda idx: [_b(g.to_effective(k)) for k in idx]
    for name in ("energy", "bass", "onsets", "tension", "vocal"):
        if name not in what:
            continue
        if name == "vocal":
            src = vocal["curve"] if vocal else None
        elif name == "tension":
            src = d.get("tension", {}).get("curve") or None
        else:
            src = st.get(CURVES[name]) or None
        if src is None:
            out[name] = {"available": False, "reason": "needs separate_stems or a v4 analysis" if name == "vocal" else "not in this analysis"}
            continue
        idx = [k for k in ks if k < len(src)]
        vals, bars, step = _downsample([src[k] for k in idx], idx, max_points)
        out[name] = {"bars": bars_of(bars), "values": vals, "step_bars": step}
        if name == "vocal":
            out[name].update(confidence=vocal["confidence"], confidence_components=vocal["confidence_components"])
    if "beats" in what:
        p = 60.0 / g.bpm
        j0, j1 = math.ceil((g.sec(start) - g.fd) / p - 1e-9), math.ceil((g.sec(end) - g.fd) / p - 1e-9)
        js = list(range(j0, j1))
        downbeats = [j for j in js if j % 4 == 0]
        shown = js if len(js) <= MAX_BEATS else downbeats
        out["beats"] = {"bpm": round(g.bpm, 3), "times_sec": [round(g.fd + j * p, 3) for j in shown],
                        "downbeats_only": shown is downbeats and len(js) > MAX_BEATS,
                        "n_beats": len(js), "confidence": t["confidence"], "downbeat_confidence": t.get("downbeat_confidence")}
    over = lambda x0, x1: x0 < b - 1e-6 and x1 > a + 1e-6
    if "sections" in what:
        out["sections"] = [{"label": s["label"], "start_bar": _b(g.to_effective(s["start_bar"])), "end_bar": _b(g.to_effective(s["end_bar"])),
                            "energy": s["energy"], "bass": s["bass"], "confidence": s["confidence"]}
                           for s in st["sections"] if over(s["start_bar"], s["end_bar"])]
    if "loops" in what:
        out["loops"] = [{**{k: v for k, v in c.items() if k != "start_bar"}, "start_bar": _b(g.to_effective(c["start_bar"]))}
                        for c in st["loop_candidates"] if over(c["start_bar"], c["start_bar"] + c["bars"])]
    if "harmony" in what:
        chroma = d.get("harmony", {}).get("chroma_bar", [])
        idx = [k for k in ks if k < len(chroma)]
        step = max(math.ceil(len(idx) / max_points), 1)
        out["harmony"] = {"bars": bars_of(idx[::step]), "step_bars": step,
                          "chroma": [np.round(np.mean([chroma[k] for k in idx[i:i + step]], axis=0), 3).tolist() for i in range(0, len(idx), step)],
                          "key_by_phrase": [{**k, "start_bar": _b(g.to_effective(k["start_bar"]))} for k in d["harmony"]["key_by_phrase"]
                                            if over(k["start_bar"], k["start_bar"] + k["bars"])]}
    if "cues" in what:
        mine = {k: v for k, v in (cues or {}).items() if start <= v["bar"] < end}
        out["cues"] = {"hot": mine, "suggested": [{**c, "bar": _b(g.to_effective(c["bar"]))} for c in d["mix_points"]["cue_points"]
                                                  if a <= c["bar"] < b]}
    if "waveform" in what and waveform:
        pts, dur = waveform["points"], d["duration_sec"]
        i0, i1 = max(int(g.sec(start) / dur * pts), 0), min(int(math.ceil(g.sec(end) / dur * pts)), pts)
        step = max(math.ceil((i1 - i0) / max_points), 1)
        out["waveform"] = {k: waveform[k][i0:i1:step] for k in ("peak", "low", "mid", "high")}
        out["waveform"].update(points=len(out["waveform"]["peak"]), step_points=step)
    return out


def check_what(what) -> tuple[str, ...]:
    bad = [w for w in what if w not in WHAT]
    if bad:
        raise OpenMixingError("invalid_value", f"unknown region field(s) {bad}", "what", f"choose from {list(WHAT)}")
    return tuple(what)
