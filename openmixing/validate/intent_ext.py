from __future__ import annotations

from .harmony import DISSONANT

CLASH_SEC = 1.0


def evaluate_ext(intent: dict, checks: dict, add) -> None:
    vc, hc = checks.get("vocal_collision"), checks.get("harmonic_clash")
    if "vocal_overlap_max_sec" in intent:
        add("vocal_overlap_max_sec", intent["vocal_overlap_max_sec"], vc["seconds_overlap"], vc["seconds_overlap"] <= intent["vocal_overlap_max_sec"],
            f"method: {vc['method']}")
    if "vocal_clash_intended" in intent:
        happened = vc["seconds_overlap"] >= CLASH_SEC
        add("vocal_clash_intended", intent["vocal_clash_intended"], vc["seconds_overlap"], happened == bool(intent["vocal_clash_intended"]),
            "intended clash happened" if happened and intent["vocal_clash_intended"] else f"method: {vc['method']}")
    dis = hc["mean_dissonance"] if hc else None
    if "harmonic_dissonance_max" in intent:
        add("harmonic_dissonance_max", intent["harmonic_dissonance_max"], dis, dis is None or dis <= intent["harmonic_dissonance_max"],
            "no measurable overlap" if dis is None else f"confidence {hc['confidence']}")
    if "harmonic_clash_intended" in intent:
        happened = dis is not None and dis >= DISSONANT
        add("harmonic_clash_intended", intent["harmonic_clash_intended"], dis, happened == bool(intent["harmonic_clash_intended"]),
            "intended clash happened" if happened and intent["harmonic_clash_intended"] else "")
    if "tension_target" in intent:
        t = checks["tension_match"]
        want = intent["tension_target"].get("min_correlation", 0.8)
        corr = t["correlation"]
        add("tension_target", {"target": t["target"], "min_correlation": want}, corr, corr is not None and corr >= want,
            f"confidence {t['confidence']}")
    if "pre_drop_gap_min_beats" in intent:
        gaps = [g["beats"] for g in checks["pre_drop_gap"]]
        best = max(gaps) if gaps else 0.0
        add("pre_drop_gap_min_beats", intent["pre_drop_gap_min_beats"], best, best >= intent["pre_drop_gap_min_beats"],
            "" if gaps else "no gap found")
