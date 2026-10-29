from __future__ import annotations

from typing import Literal

from ... import candidates, mixing
from ..app import tool
from ..state import lib
from ..versions import versions
from ..views import jsonable


@tool()
def compare_tracks(from_track: str, to_track: str) -> dict:
    """Mixability of from_track -> to_track: key relation, tempo stretch needed (half/double time aware), energy delta, score 0-100,
    and a suggested transition style / length."""
    a, b = lib().effective(from_track), lib().effective(to_track)
    c = mixing.compat(a, b)
    k, tp = c["key"]["distance"], abs(c["tempo"]["stretch_percent"])
    if tp > 8:
        sug = {"style": "echo_out", "bars": 4, "why": "tempos too far apart to beatmatch cleanly"}
    elif k <= 1 and tp <= 4:
        sug = {"style": "long_blend", "bars": 32, "why": "harmonic match and close tempo: blend long"}
    elif k <= 2:
        sug = {"style": "bass_swap", "bars": 16, "why": "compatible keys: classic bass-swap blend"}
    else:
        sug = {"style": "filter_fade", "bars": 8, "why": "keys clash: keep the overlap short and filtered"}
    return {"from": a["file"], "to": b["file"], **c, "suggested_transition": sug}


@tool()
def suggest_next(track: str, count: int = 8, max_stretch_percent: float = 8.0, min_score: float = 0.0) -> dict:
    """Rank analyzed tracks to follow `track` (harmonic + tempo + energy)."""
    a = lib().effective(track)
    rows = []
    for t in lib().scan():
        if t["id"] == a["id"] or not lib().cached(t):
            continue
        b = lib().effective(t["id"])
        c = mixing.compat(a, b)
        if abs(c["tempo"]["stretch_percent"]) <= max_stretch_percent and c["score"] >= min_score:
            rows.append({"id": b["id"], "file": b["file"], "score": c["score"], "key": f"{b['camelot']} ({c['key']['relation']})",
                         "bpm": round(b["bpm"], 2), "stretch_percent": c["tempo"]["stretch_percent"], "energy": b["energy"]})
    return {"after": a["file"], "candidates": sorted(rows, key=lambda r: -r["score"])[:count]}


@tool()
def suggest_order(tracks: list[str] | None = None, strategy: Literal["harmonic", "energy_build", "bpm_ramp"] = "harmonic",
                  start_with: str | None = None) -> dict:
    """Order tracks (default: all analyzed) into a set: harmonic = best pairwise compatibility, energy_build = rising energy,
    bpm_ramp = rising tempo. Fast greedy path; build_set is the full builder (arcs, constraints, trimming, a plan)."""
    refs = tracks or [t["id"] for t in lib().scan() if lib().cached(t)]
    effs = [lib().effective(r) for r in refs]
    path = mixing.order_tracks(effs, strategy, lib().resolve(start_with)["id"] if start_with else None)
    steps = [{"track": e["file"], "bpm": round(e["bpm"], 2), "camelot": e["camelot"], "energy": e["energy"]} for e in path]
    for i in range(1, len(path)):
        c = mixing.compat(path[i - 1], path[i])
        steps[i]["from_previous"] = {"score": c["score"], "key": c["key"]["relation"], "stretch_percent": c["tempo"]["stretch_percent"]}
    return {"strategy": strategy, "order": steps}


@tool()
def find_windows(track: str, side: Literal["exit", "entry"], limit: int = 5) -> dict:
    """Usable regions to leave (side="exit") or enter (side="entry") a track: outros, intros, breakdowns, quiet low-bass sections.
    Each window has start/end bar, label, energy, bass, vocal presence (needs stems), phrase alignment, score and confidence."""
    f = candidates.track_facts(lib(), track)
    ve = f["vocal_entry"]
    return jsonable({"track": f["e"]["file"], "side": side, "phrase_bars": f["phrase"], "n_bars": f["n_bars"],
                     "tempo_confidence": f["tempo_conf"], "key_confidence": f["key_conf"],
                     "vocal": {"available": bool(ve), "confidence": ve["confidence"] if ve else 0.0,
                               "confidence_components": ve["confidence_components"] if ve else None},
                     "windows": candidates.find_windows(f, side)[:limit], "versions": versions()})


@tool()
def pair_report(a: str, b: str, exit_window: str | None = None, entry_window: str | None = None) -> dict:
    """Measurements for leaving track `a` into track `b`: tempo ratio and % stretch (half/double aware), key relation and semitones
    to match, loudness difference, low-end and vocal overlap risk, energy and phrase alignment. Windows default to the best ones
    (ids from find_windows). Every block carries a confidence; nothing is judged."""
    fa, fb = candidates.track_facts(lib(), a), candidates.track_facts(lib(), b)
    pick = lambda f, side, wid: next((w for w in candidates.find_windows(f, side) if wid is None or w["id"] == wid), None)
    wa, wb = pick(fa, "exit", exit_window), pick(fb, "entry", entry_window)
    for w, wid in ((wa, exit_window), (wb, entry_window)):
        if w is None:
            raise ValueError(f"no window {wid!r}; call find_windows")
    return jsonable({**candidates.pair_report(fa, fb, wa, wb), "versions": versions()})


@tool()
def suggest_transitions(a: str, b: str, goals: list[str] | dict[str, float] | None = None, limit: int = 5, include_plan: bool = True) -> dict:
    """Several feasible transitions from `a` into `b`, ranked for the goals (smooth, energetic, dramatic, experimental; default smooth;
    a dict weights them). Each candidate has sub-scores (tempo, harmony, energy_flow, bass_clash, vocal_clash, phrase, impact,
    novelty; 1 = good), a confidence, and a plan-schema document for just these two clips (apply_plan replaces the whole set: merge it into
    export_plan output first).
    Infeasible techniques are listed with the reason (e.g. stems missing). include_plan=False returns just the scores."""
    r = candidates.suggest(lib(), a, b, goals, limit)
    r["versions"] = versions()
    if not include_plan:
        for c in r["candidates"]:
            c.pop("plan")
    return jsonable(r)
