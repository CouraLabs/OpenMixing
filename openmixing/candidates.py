"""Evidence for planning a transition: usable windows, pair measurements and scored candidate plans.

Everything here measures; the sub-scores rank candidates for a goal, they never veto one. Candidate plans are
plan-schema documents (see plan.py) built by the planner templates and checked with validate_plan.
"""
from __future__ import annotations

import json
import math

import numpy as np
import soundfile as sf

from . import plan as planlib
from .analysis.confidence import combine, unit
from .planner import TEMPLATES, apply_template, camelot_relation, tempo_match
from .setmodel import Clip, SetModel

BAR = 4.0
VERSION = 2
MAX_OVERLAP_BARS = 32

GOALS = {  # weight per sub-score
    "smooth": {"tempo": 3, "harmony": 3, "energy_flow": 2, "bass_clash": 3, "vocal_clash": 2, "phrase": 2, "blend": 3, "impact": 0, "novelty": 0},
    "energetic": {"tempo": 2, "harmony": 1, "energy_flow": 1, "bass_clash": 2, "vocal_clash": 1, "phrase": 2, "blend": 1, "impact": 3, "novelty": 0},
    "dramatic": {"tempo": 1, "harmony": 1, "energy_flow": 0.5, "bass_clash": 1, "vocal_clash": 1, "phrase": 2, "blend": 0, "impact": 4, "novelty": 1},
    "experimental": {"tempo": 0.5, "harmony": 0.5, "energy_flow": 0, "bass_clash": 1, "vocal_clash": 0.5, "phrase": 1, "blend": 0, "impact": 2, "novelty": 4},
}
# Style properties (heuristic, 0..1): how long the harmony / bass of both tracks is exposed together, how much of the handover
# is a true overlap (blend), punch, unusualness.
STYLES = {
    "bass_swap": {"blend": 0.85, "harmony": 0.8, "bass": 0.3, "impact": 0.35, "novelty": 0.1},
    "long_blend": {"blend": 1.0, "harmony": 1.0, "bass": 0.4, "impact": 0.2, "novelty": 0.0},
    "filter_fade": {"blend": 0.8, "harmony": 0.6, "bass": 0.35, "impact": 0.5, "novelty": 0.3},
    "echo_out": {"blend": 0.45, "harmony": 0.25, "bass": 0.3, "impact": 0.7, "novelty": 0.5},
    "drop_swap": {"blend": 0.1, "harmony": 0.0, "bass": 0.0, "impact": 0.9, "novelty": 0.4},
    "loop_roll_out": {"blend": 0.0, "harmony": 0.1, "bass": 0.2, "impact": 0.8, "novelty": 0.7},
    "stem_blend": {"blend": 0.9, "harmony": 0.5, "bass": 0.0, "impact": 0.4, "novelty": 0.8},
}
MIN_BARS = {"bass_swap": 8, "long_blend": 16, "filter_fade": 8, "echo_out": 4, "drop_swap": 1, "loop_roll_out": 2, "stem_blend": 8}


def _r(x, n=3):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- per-track evidence
def _vocal_file(lib, tid: str):
    return lib.state / "analysis" / f"{tid}.vocal.json"


def vocal_activity(y: np.ndarray, sr: int, first_downbeat: float, bpm: float, n_bars: int) -> list[float]:
    """Per-bar vocal presence 0..1 from a vocals stem: bar RMS relative to the loudest bar, silent below -35 dB."""
    bar = 240.0 / bpm
    rms = []
    for k in range(n_bars):
        a, b = int((first_downbeat + k * bar) * sr), int((first_downbeat + (k + 1) * bar) * sr)
        seg = y[a:b]
        rms.append(float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0)
    db = 20 * np.log10(np.maximum(np.array(rms), 1e-9) / max(max(rms, default=0), 1e-9))
    return [round(float(v), 3) for v in np.clip((db + 35) / 35, 0, 1)]


def vocal_entry(lib, e: dict, n_bars: int) -> dict | None:
    """{curve, confidence, confidence_components} of vocal presence per bar when the track's stems exist (cached), else None."""
    if not e.get("stems"):
        return None
    f = _vocal_file(lib, e["id"])
    key = [e["bpm"], e["first_downbeat"], n_bars]
    try:
        hit = json.loads(f.read_text())
        if hit["key"] == key and "confidence" in hit:
            return hit
    except (OSError, ValueError, KeyError):
        pass
    y, sr = sf.read(e["stems"]["vocals"], dtype="float32", always_2d=True)
    y = y.mean(1)
    curve = vocal_activity(y, sr, e["first_downbeat"], e["bpm"], n_bars)
    peak_db = 20 * np.log10(max(float(np.sqrt((y ** 2).mean())), 1e-9))
    c = np.array(curve) if curve else np.zeros(1)
    conf, comps = combine({"stem_level": unit(peak_db, -50, -20), "contrast": float(np.percentile(c, 90) - np.percentile(c, 10))})
    hit = {"key": key, "curve": curve, "confidence": conf, "confidence_components": comps}
    f.write_text(json.dumps(hit))
    return hit


def vocal_curve(lib, e: dict, n_bars: int) -> list[float] | None:
    """Vocal presence per bar when the track's stems exist (cached), else None."""
    hit = vocal_entry(lib, e, n_bars)
    return hit["curve"] if hit else None


def track_facts(lib, ref: str) -> dict:
    e = lib.effective(ref)
    d = lib.analyze(e["id"])
    st, t = d["structure"], d["tempo"]
    n_bars = int(t["n_bars"]) if "bpm" not in e["overrides"] else int((e["duration"] - e["first_downbeat"]) * e["bpm"] / 240)
    ov = e["overrides"]
    ve = vocal_entry(lib, e, n_bars)
    return {"e": e, "d": d, "n_bars": n_bars, "vocal_entry": ve, "sections": st["sections"], "phrase": st["phrase_bars"] or 16,
            "energy": st["energy_curve"], "bass": st["bass_curve"], "vocal": ve["curve"] if ve else None,
            "tempo_conf": 1.0 if "bpm" in ov else round(t["confidence"], 3),
            "key_conf": 1.0 if ("camelot" in ov or "key" in ov) else round(d["key"]["confidence"], 3)}


def _mean(curve, a: int, b: int):
    seg = curve[max(a, 0): max(b, 0)]
    return float(np.mean(seg)) if len(seg) else None


# ---------------------------------------------------------------- windows
def find_windows(f: dict, side: str) -> list[dict]:
    """Usable regions to leave (`exit`) or enter (`entry`) a track, best first. `score` ranks blend suitability (quiet, low bass, phrase aligned)."""
    if side not in ("exit", "entry"):
        raise ValueError("side must be 'exit' or 'entry'")
    secs, n, ph = f["sections"], f["n_bars"], f["phrase"]
    prefer = ("outro", "breakdown", "groove", "build") if side == "exit" else ("intro", "breakdown", "groove", "build")
    base_conf = min(f["tempo_conf"], 1.0)
    raw = []
    long_track = n > 2 * ph
    for i, s in enumerate(secs):
        if s["label"] == "drop" and side == "entry":
            continue
        if long_track and ((side == "exit" and (s["label"] == "intro" or s["start_bar"] < ph))
                           or (side == "entry" and (s["label"] == "outro" or s["start_bar"] >= n - ph))):
            continue  # never leave a track from its first phrase, nor enter it through its last
        a, b = s["start_bar"], s["end_bar"]
        edge = (i == len(secs) - 1) if side == "exit" else (i == 0)
        label_conf = 0.85 if (s["label"] in ("intro", "outro") and edge) else 0.6
        raw.append((s["label"], a, b, label_conf, s["energy"], s["bass"]))
    cue = f["d"]["mix_points"]["mix_out" if side == "exit" else "mix_in"]
    if side == "exit":
        a = max(int(cue["bar"]), ph if long_track else 0)
        raw.append(("mix_out", a, n, 0.7, _mean(f["energy"], a, n), _mean(f["bass"], a, n)))
    else:
        b = min(n, max(ph, 8))
        raw.append(("mix_in", 0, b, 0.7, _mean(f["energy"], 0, b), _mean(f["bass"], 0, b)))
    out, seen = [], set()
    for label, a, b, lconf, en, bs in raw:
        if b - a < 4 or (a, b) in seen:
            continue
        seen.add((a, b))
        vox = _mean(f["vocal"], a, b) if f["vocal"] else None
        aligned = a % ph == 0
        quiet = 1 - (0.5 * (en or 0) + 0.5 * (bs or 0))
        score = 0.5 * quiet + 0.2 * aligned + 0.2 * (label in prefer[:1]) + 0.1 * (1 - (vox or 0))
        out.append({"id": f"{side}:{label}@{a}", "side": side, "label": label, "start_bar": a, "end_bar": b, "bars": b - a,
                    "energy": _r(en), "bass": _r(bs), "vocal": _r(vox), "phrase_aligned": bool(aligned),
                    "score": _r(score), "confidence": _r(base_conf * lconf),
                    "components": {"tempo_grid": f["tempo_conf"], "label": lconf, "phrase_bars": ph}})
    if not out:  # too short for sections: the last / first phrase
        w = min(ph, n)
        a = max(n - w, 0) if side == "exit" else 0
        out.append({"id": f"{side}:{'tail' if side == 'exit' else 'head'}@{a}", "side": side, "label": "tail" if side == "exit" else "head",
                    "start_bar": a, "end_bar": a + w, "bars": w, "energy": None, "bass": None, "vocal": None,
                    "phrase_aligned": a % ph == 0, "score": 0.3, "confidence": _r(base_conf * 0.4),
                    "components": {"tempo_grid": f["tempo_conf"], "label": 0.4, "phrase_bars": ph}})
    return sorted(out, key=lambda w: -w["score"])


# ---------------------------------------------------------------- pair report
_MAJ_PC = {n: ((n - 8) * 7) % 12 for n in range(1, 13)}  # Camelot number -> major-key pitch class (8B = C)


def key_shift_semitones(cam_from: str, cam_to: str) -> int:
    """Semitones (-6..6) to move the `to` track so it sits in the `from` key (relative keys count as the same)."""
    d = (_MAJ_PC[int(cam_from[:-1])] - _MAJ_PC[int(cam_to[:-1])]) % 12  # a minor key shares its relative major's number
    return d - 12 if d > 6 else d


def _extra_bars(curve, start: int, step: int, ref: float | None, limit: int, lo: int, hi: int, slack: float = 0.25) -> int:
    """Bars the window can grow from `start` in direction `step` while the track's energy stays within `slack` of the window's."""
    ref = 0.5 if ref is None else ref
    n, i = 0, start
    while n < limit and lo <= i < hi and (not curve or curve[i] <= ref + slack):
        n, i = n + 1, i + step
    return n


def extension(fa: dict, fb: dict, wa: dict, wb: dict) -> dict:
    """How far the transition may run past the edges of both windows into adjacent sections of similar or lower energy."""
    ea = wa["energy"] if wa["energy"] is not None else None
    eb = wb["energy"] if wb["energy"] is not None else None
    xa = _extra_bars(fa["energy"], wa["start_bar"] - 1, -1, ea, MAX_OVERLAP_BARS, 0, fa["n_bars"])
    xb = _extra_bars(fb["energy"], wb["end_bar"], 1, eb, MAX_OVERLAP_BARS, 0, fb["n_bars"])
    room = int(min(wa["bars"] + xa, wb["bars"] + xb, MAX_OVERLAP_BARS))
    return {"a_extra_bars": xa, "b_extra_bars": xb, "room_with_extension": room}


def pair_report(fa: dict, fb: dict, wa: dict | None = None, wb: dict | None = None, bars: int | None = None) -> dict:
    """Measurements for A leaving through window `wa` into B entering through `wb` (defaults: best of each).
    With `bars`, bass / vocal overlap are measured over that exact overlap (A's last `bars` bars before the window end, B's first
    `bars` from the window start) instead of the full window."""
    wa = wa or find_windows(fa, "exit")[0]
    wb = wb or find_windows(fb, "entry")[0]
    ea, eb = fa["e"], fb["e"]
    tm = tempo_match(ea["bpm"], eb["bpm"])
    tconf = min(fa["tempo_conf"], fb["tempo_conf"])
    pitch_st = 12 * math.log2(1 + tm["stretch_percent"] / 100) if tm["stretch_percent"] > -100 else 0.0
    dist, rel = camelot_relation(ea["camelot"], eb["camelot"])
    kconf = min(fa["key_conf"], fb["key_conf"])
    la, lb = fa["d"]["loudness"], fb["d"]["loudness"]
    L = int(min(wa["bars"], wb["bars"], MAX_OVERLAP_BARS))
    ext = extension(fa, fb, wa, wb)
    if bars:
        L = int(bars)
    a0 = max(wa["end_bar"] - L, 0)
    ba = fa["bass"][a0: wa["end_bar"]] if fa["bass"] else []
    bb = fb["bass"][wb["start_bar"]: wb["start_bar"] + L] if fb["bass"] else []
    n = min(len(ba), len(bb))
    bass_risk = float(np.mean(np.minimum(ba[:n], bb[:n]))) if n else None
    if fa["vocal"] is not None and fb["vocal"] is not None:
        va, vb = fa["vocal"][a0: wa["end_bar"]], fb["vocal"][wb["start_bar"]: wb["start_bar"] + L]
        m = min(len(va), len(vb))
        vox_risk = float(np.mean(np.minimum(va[:m], vb[:m]))) if m else None
        vox = {"risk": _r(vox_risk), "a_presence": _r(np.mean(va)) if len(va) else None, "b_presence": _r(np.mean(vb)) if len(vb) else None,
               "confidence": 0.7, "basis": "vocal stem RMS per bar over the overlap (min of both)"}
    else:
        vox = {"risk": None, "confidence": 0.0, "basis": "needs separate_stems on both tracks"}
    ea_en, eb_en = _mean(fa["energy"], wa["start_bar"], wa["end_bar"]), _mean(fb["energy"], wb["start_bar"], wb["end_bar"])
    return {
        "windows": {"exit": wa["id"], "entry": wb["id"], "overlap_bars_max": int(min(wa["bars"], wb["bars"], MAX_OVERLAP_BARS)), "overlap_bars_measured": L, **ext},
        "tempo": {**tm, "pitch_shift_semitones_if_varispeed": _r(pitch_st), "confidence": _r(tconf),
                  "components": {"a_grid": fa["tempo_conf"], "b_grid": fb["tempo_conf"]}},
        "key": {"from": ea["camelot"], "to": eb["camelot"], "distance": dist, "relation": rel,
                "semitones_to_match": key_shift_semitones(ea["camelot"], eb["camelot"]), "confidence": _r(kconf),
                "components": {"a": fa["key_conf"], "b": fb["key_conf"]}},
        "loudness": {"a_lufs": la["integrated_lufs"], "b_lufs": lb["integrated_lufs"],
                     "difference_lu": _r(lb["integrated_lufs"] - la["integrated_lufs"], 2), "confidence": 0.9},
        "low_end": {"overlap_risk": _r(bass_risk), "a_bass": _r(np.mean(ba)) if len(ba) else None, "b_bass": _r(np.mean(bb)) if len(bb) else None,
                    "confidence": 0.6, "basis": "bar-level low-band (<150 Hz) energy (0..1) of the full mix of both tracks over the overlap; min() of the two",
                    "stem_low_end_carrier": {"a": ea.get("bass_carrier"), "b": eb.get("bass_carrier")}},
        "vocal": vox,
        "energy": {"a": _r(ea_en), "b": _r(eb_en), "delta": _r(eb_en - ea_en) if ea_en is not None and eb_en is not None else None, "confidence": 0.6},
        "phrase": {"a_exit_bar": wa["end_bar"], "a_phrase_bars": fa["phrase"], "a_aligned": wa["end_bar"] % fa["phrase"] == 0 or wa["end_bar"] == fa["n_bars"],
                   "b_entry_bar": wb["start_bar"], "b_phrase_bars": fb["phrase"], "b_aligned": wb["start_bar"] % fb["phrase"] == 0, "confidence": 0.7},
    }


# ---------------------------------------------------------------- candidates
def resolve_goals(goals) -> dict[str, float]:
    if not goals:
        goals = {"smooth": 1.0}
    if isinstance(goals, str):
        goals = [goals]
    if not isinstance(goals, dict):
        goals = {g: 1.0 for g in goals}
    bad = [g for g in goals if g not in GOALS]
    if bad:
        raise ValueError(f"unknown goals {bad}; choose from {list(GOALS)}")
    w: dict[str, float] = {}
    for g, k in goals.items():
        for s, v in GOALS[g].items():
            w[s] = w.get(s, 0.0) + v * float(k)
    return w


def sub_scores(rep: dict, style: str, bars: int, at_a: int) -> dict:
    sp = STYLES[style]
    expo = min(max(0.5 + bars / 32, 0.5), 1.5)  # longer overlap = longer exposure to the same clash
    stretch = abs(rep["tempo"]["stretch_percent"])
    tempo = max(0.0, 1 - stretch / 8)
    base_h = {0: 1.0, 1: 0.9, 2: 0.6, 3: 0.3}.get(rep["key"]["distance"], 0.1)
    harmony = 1 - min(sp["harmony"] * expo, 1.0) * (1 - base_h)
    d = rep["energy"]["delta"]
    flow = 0.7 if d is None else 1 - min(abs(d), 1.0)
    br = rep["low_end"]["overlap_risk"]
    bass = 0.7 if br is None else 1 - br * sp["bass"] * 2 * expo
    vr = rep["vocal"]["risk"]
    vocal = 1.0 if vr is None else 1 - vr * {"stem_blend": 0.5, "drop_swap": 0.0}.get(style, 1.0) * expo
    ph = rep["phrase"]
    phrase = 0.35 * ph["a_aligned"] + 0.35 * ph["b_aligned"] + 0.15 * (at_a % 4 == 0) + 0.15 * (bars % 4 == 0)
    impact = sp["impact"] if d is None else min(1.0, sp["impact"] * 0.8 + 0.2 * (0.5 + 0.5 * max(min(d, 1), -1)))
    return {"tempo": tempo, "harmony": harmony, "energy_flow": flow, "bass_clash": max(bass, 0.0), "vocal_clash": max(vocal, 0.0),
            "phrase": phrase, "blend": sp["blend"] * min(1.0, 0.5 + bars / 32), "impact": impact, "novelty": sp["novelty"]}


def _confidence(rep: dict, weights: dict) -> float:
    c = {"tempo": rep["tempo"]["confidence"], "harmony": rep["key"]["confidence"], "bass_clash": rep["low_end"]["confidence"],
         "vocal_clash": rep["vocal"]["confidence"] or 1.0, "energy_flow": rep["energy"]["confidence"], "phrase": rep["phrase"]["confidence"]}
    used = [(c[k], weights.get(k, 0)) for k in c if weights.get(k, 0) > 0]
    tot = sum(x for _, x in used) or 1.0
    return _r(sum(a * x for a, x in used) / tot)


def _fragment(lib, fa: dict, fb: dict, wa: dict, wb: dict, style: str, bars: int) -> tuple[dict, int]:
    """Plan document for `style` leaving A at the end of window `wa` and entering B at the start of `wb`."""
    at = max(wa["end_bar"] - bars, 0)
    lead = min(16, at)
    out = Clip("out", fa["e"]["id"], (at - lead) * BAR, (lead + bars) * BAR, label="outgoing")
    tail = max(min(16, fb["n_bars"] - wb["start_bar"] - bars), 0)
    inc = Clip("in", fb["e"]["id"], wb["start_bar"] * BAR, (bars + tail) * BAR, label="incoming")
    if wb["start_bar"] + bars + tail > fb["n_bars"] or at - lead + lead + bars > fa["n_bars"]:
        raise ValueError("window runs past the end of a track")
    m = SetModel(f"transition_{style}", [out, inc])
    apply_template(m, lib.render_info, out, inc, style, float(lead), float(bars), trim_out=True)
    for c in (out, inc):  # ramps that end past the clip's last beat (echo_out, drop_swap) are cut at the end
        for lane in list(c.lanes.values()) + [c.tempo]:
            if lane.points and lane.points[-1][0] > c.length_beats:
                end = float(lane.value_at(c.length_beats))
                lane.points = [p for p in lane.points if p[0] < c.length_beats] + [[c.length_beats, end]]
    return planlib.export_plan(m), at


def suggest(lib, a: str, b: str, goals=None, limit: int = 5, max_windows: int = 2) -> dict:
    fa, fb = track_facts(lib, a), track_facts(lib, b)
    weights = resolve_goals(goals)
    tot_w = sum(weights.values()) or 1.0
    wins_a, wins_b = find_windows(fa, "exit")[:max_windows], find_windows(fb, "entry")[:max_windows]
    cands, infeasible = [], []
    for wa in wins_a:
        for wb in wins_b:
            rep0 = pair_report(fa, fb, wa, wb)
            win = rep0["windows"]
            base_room, room = win["overlap_bars_max"], win["room_with_extension"]
            sizes = sorted({s for s in (8, 16, 32, base_room, room) if 1 <= s <= room})
            for style in TEMPLATES:
                why = None
                if style == "stem_blend" and not (fa["e"].get("stems") and fb["e"].get("stems")):
                    why = "stems missing: call separate_stems on both tracks"
                elif style == "long_blend" and rep0["key"]["distance"] > 1:
                    why = f"long_blend needs a harmonic match (Camelot distance {rep0['key']['distance']})"
                elif room < MIN_BARS[style]:
                    why = (f"window gives {base_room} bars ({room} by extending {win['a_extra_bars']} bars earlier in A / "
                           f"{win['b_extra_bars']} later in B while energy stays similar), {style} needs {MIN_BARS[style]}")
                if why:
                    infeasible.append({"style": style, "windows": [wa["id"], wb["id"]], "reason": why})
                    continue
                use = [s for s in sizes if s >= MIN_BARS[style]] or [room]
                for bars in (use if style not in ("drop_swap", "loop_roll_out", "echo_out") else use[:1]):
                    try:
                        frag, at = _fragment(lib, fa, fb, wa, wb, style, bars)
                        errs = planlib.validate_plan(frag, lib)
                        if errs:
                            raise ValueError(errs[0]["message"])
                    except ValueError as e:
                        infeasible.append({"style": style, "windows": [wa["id"], wb["id"]], "reason": str(e)})
                        continue
                    rep = pair_report(fa, fb, wa, wb, bars)
                    sc = sub_scores(rep, style, bars, at)
                    total = sum(sc[k] * weights.get(k, 0) for k in sc) / tot_w
                    cands.append({"style": style, "windows": [wa["id"], wb["id"]], "bars": bars, "exit_bar_in_a": at,
                                  "total": _r(total * 100, 1), "sub_scores": {k: _r(v) for k, v in sc.items()},
                                  "confidence": _confidence(rep, weights), "plan": frag})
                    if bars > base_room:
                        cands[-1]["extended_bars"] = {"a": max(wa["start_bar"] - at, 0), "b": max(bars - wb["bars"], 0)}
    cands.sort(key=lambda c: -c["total"])
    seen_sc, uniq_c = set(), []
    for c in cands:  # identical styles + sub-scores are the same transition; keep the shortest
        key = (c["style"], tuple(c["sub_scores"].values()))
        if key not in seen_sc:
            seen_sc.add(key)
            uniq_c.append(c)
    cands = uniq_c
    for i, c in enumerate(cands):
        c["id"] = f"cand{i + 1}"
    uniq = {}
    for x in infeasible:
        uniq.setdefault((x["style"], x["reason"]), x)
    return {"version": VERSION, "plan_schema": planlib.PLAN_VERSION, "goals": goals or {"smooth": 1.0}, "weights": weights,
            "pair": pair_report(fa, fb, wins_a[0], wins_b[0]), "candidates": cands[:limit], "n_feasible": len(cands),
            "infeasible": list(uniq.values())[:12],
            "note": "scores rank candidates for the goal weights; sub-scores and confidence show why. Pass a candidate's plan to apply_plan."}
