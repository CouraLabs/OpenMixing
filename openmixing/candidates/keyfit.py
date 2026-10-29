from __future__ import annotations


def fit_key(rep: dict, plan: dict) -> tuple[dict, dict | None]:
    km = rep["key"]["match"]
    if km["low_confidence"] or km["distance_after"] >= km["distance_before"] or not km["semitones_in"]:
        return rep, None
    for c in plan["clips"]:
        if c["id"] == "in":
            c["semitones"] = km["semitones_in"]
    fitted = {**rep, "key": {**rep["key"], "distance": km["distance_after"]}}
    return fitted, {"semitones": km["semitones_in"], "camelot": km["camelot_after"], "distance_before": km["distance_before"]}
