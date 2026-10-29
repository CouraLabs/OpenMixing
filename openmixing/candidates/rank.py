from __future__ import annotations

from .. import plan as planlib
from ..core.units import BAR
from ..mixing.transitions import TEMPLATES, apply_template, clamp_to_clip
from ..model.clip import Clip
from ..model.setmodel import SetModel
from .keyfit import fit_key
from .pair import pair_report
from .windows import find_windows, rnd, track_facts

VERSION = 2




GOALS = {
    "smooth": {"tempo": 3, "harmony": 3, "energy_flow": 2, "bass_clash": 3, "vocal_clash": 2, "phrase": 2, "blend": 3, "impact": 0, "novelty": 0},
    "energetic": {"tempo": 2, "harmony": 1, "energy_flow": 1, "bass_clash": 2, "vocal_clash": 1, "phrase": 2, "blend": 1, "impact": 3, "novelty": 0},
    "dramatic": {"tempo": 1, "harmony": 1, "energy_flow": 0.5, "bass_clash": 1, "vocal_clash": 1, "phrase": 2, "blend": 0, "impact": 4, "novelty": 1},
    "experimental": {"tempo": 0.5, "harmony": 0.5, "energy_flow": 0, "bass_clash": 1, "vocal_clash": 0.5, "phrase": 1, "blend": 0, "impact": 2,
                     "novelty": 4},
}


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
    expo = min(max(0.5 + bars / 32, 0.5), 1.5)
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
    return rnd(sum(a * x for a, x in used) / tot)


def _fragment(lib, fa: dict, fb: dict, wa: dict, wb: dict, style: str, bars: int) -> tuple[dict, int]:
    at = max(wa["end_bar"] - bars, 0)
    lead = min(16, at)
    out = Clip("out", fa["e"]["id"], (at - lead) * BAR, (lead + bars) * BAR, label="outgoing")
    tail = max(min(16, fb["n_bars"] - wb["start_bar"] - bars), 0)
    inc = Clip("in", fb["e"]["id"], wb["start_bar"] * BAR, (bars + tail) * BAR, label="incoming")
    if wb["start_bar"] + bars + tail > fb["n_bars"] or at - lead + lead + bars > fa["n_bars"]:
        raise ValueError("window runs past the end of a track")
    m = SetModel(f"transition_{style}", [out, inc])
    apply_template(m, lib.effective, out, inc, style, float(lead), float(bars), trim_out=True)
    clamp_to_clip(out, inc)
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
                    rep, shifted = fit_key(pair_report(fa, fb, wa, wb, bars), frag)
                    sc = sub_scores(rep, style, bars, at)
                    total = sum(sc[k] * weights.get(k, 0) for k in sc) / tot_w
                    cands.append({"style": style, "windows": [wa["id"], wb["id"]], "bars": bars, "exit_bar_in_a": at,
                                  "total": rnd(total * 100, 1), "sub_scores": {k: rnd(v) for k, v in sc.items()},
                                  "confidence": _confidence(rep, weights), "plan": frag})
                    if shifted:
                        cands[-1]["key_match"] = shifted
                    if bars > base_room:
                        cands[-1]["extended_bars"] = {"a": max(wa["start_bar"] - at, 0), "b": max(bars - wb["bars"], 0)}
    cands.sort(key=lambda c: -c["total"])
    seen_sc, uniq_c = set(), []
    for c in cands:
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
