from __future__ import annotations

from .intent_ext import evaluate_ext
from .metrics import DRIFT_OK_MS

INTENT_KEYS = ("beat_drift_ms_max", "beat_drift_intended", "loudness_jump_lu", "silence_intended", "low_end_overlap_max_sec",
               "low_end_clash_intended", "true_peak_max_dbtp", "vocal_overlap_max_sec", "vocal_clash_intended",
               "harmonic_dissonance_max", "harmonic_clash_intended", "tension_target", "pre_drop_gap_min_beats")


def intent_report(intent: dict, checks: dict) -> list[dict]:
    out = []
    add = lambda item, expected, measured, ok, note="": out.append({"item": item, "expected": expected, "measured": measured,
                                                                     "status": "pass" if ok else "fail", "note": note})
    drift = [d["max_drift_ms"] for d in checks["beat_alignment"]]
    worst = max(drift) if drift else 0.0
    if "beat_drift_ms_max" in intent:
        add("beat_drift_ms_max", intent["beat_drift_ms_max"], worst, worst <= intent["beat_drift_ms_max"])
    if "beat_drift_intended" in intent:
        happened = worst > DRIFT_OK_MS
        add("beat_drift_intended", intent["beat_drift_intended"], worst, happened == bool(intent["beat_drift_intended"]),
            "intended drift happened" if happened and intent["beat_drift_intended"] else "")
    if "loudness_jump_lu" in intent:
        lo, hi = intent["loudness_jump_lu"]
        jumps = [j["jump_lu"] for j in checks["loudness_continuity"] if j["jump_lu"] is not None]
        add("loudness_jump_lu", [lo, hi], jumps, bool(jumps) and all(lo <= j <= hi for j in jumps),
            "" if jumps else "no overlap with measurable loudness on both sides")
    if "silence_intended" in intent:
        got = [s for s in checks["planned_gaps"] + checks.get("declared_silence", [])]
        for a, b in intent["silence_intended"]:
            hit = any(s["start_sec"] < b and s["end_sec"] > a for s in got)
            add("silence_intended", [a, b], hit, hit, "" if hit else "the declared silence did not happen")
    lo_sec = checks["low_end_overlap"]["overlap_sec"]
    if "low_end_overlap_max_sec" in intent:
        add("low_end_overlap_max_sec", intent["low_end_overlap_max_sec"], lo_sec, lo_sec <= intent["low_end_overlap_max_sec"])
    if "low_end_clash_intended" in intent:
        happened = lo_sec >= 1.0
        add("low_end_clash_intended", intent["low_end_clash_intended"], lo_sec, happened == bool(intent["low_end_clash_intended"]),
            "intended clash happened" if happened and intent["low_end_clash_intended"] else "")
    if "true_peak_max_dbtp" in intent:
        tp = checks["clipping"]["true_peak_dbtp"]
        add("true_peak_max_dbtp", intent["true_peak_max_dbtp"], tp, tp is not None and tp <= intent["true_peak_max_dbtp"])
    evaluate_ext(intent, checks, add)
    return out


