from __future__ import annotations

import numpy as np
import soundfile as sf

from ..model.setmodel import SetModel
from .gap import gap_beats, silent_gaps
from .harmony import harmonic_clash
from .intent import intent_report
from .metrics import beat_drift, loudness_continuity, low_end_overlap, silences
from .tension import rendered_tension, tension_match
from .vocal import vocal_collision

VERSION = 3
TENSION_WINDOW_SEC = 1.0
def validate_render(path, model: SetModel, info, rows: list[dict], t0: float, meta: dict) -> dict:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.T
    intent = getattr(model, "intent", None) or {}
    sil = silences(x, t0, rows)
    for s_ in sil:
        if s_["planned"] is False and any(s_["start_sec"] < b and s_["end_sec"] > a for a, b in intent.get("silence_intended", [])):
            s_["planned"], s_["declared"] = True, True
    clipped = int((np.abs(x) >= 0.9999).sum())
    drift = beat_drift(model, info)
    mt = meta.get("meters") or {}
    bs = mt.get("meter_block_sec", 0.5)
    checks = {
        "clipping": {"clipped_samples": clipped, "true_peak_dbtp": meta.get("true_peak_dbtp"),
                     "ok": clipped == 0 and (meta.get("true_peak_dbtp") or -99) <= -0.1},
        "unplanned_silence": {"regions": [s for s in sil if not s["planned"]], "ok": not any(not s["planned"] for s in sil)},
        "planned_gaps": [s for s in sil if s["planned"]],
        "low_end_overlap": low_end_overlap(meta.get("low_db") or {}, meta.get("low_block_sec", 0.25), t0),
        "vocal_collision": vocal_collision(mt.get("vocal_db", {}), mt.get("vocal_method", {}), mt.get("vocal_conf", {}), bs, t0),
        "harmonic_clash": harmonic_clash(mt.get("chroma", {}), mt.get("chroma_db", {}), bs, t0),
        "beat_alignment": drift,
        "loudness_continuity": loudness_continuity(x, t0, rows),
    }
    if "tension_target" in intent or "pre_drop_gap_min_beats" in intent:
        mono = x.mean(0)
        if "tension_target" in intent:
            checks["tension_match"] = tension_match(rendered_tension(mono, sr, TENSION_WINDOW_SEC), t0, TENSION_WINDOW_SEC, intent["tension_target"])
        if "pre_drop_gap_min_beats" in intent:
            checks["pre_drop_gap"] = gap_beats(silent_gaps(mono, sr, t0), model, info, rows)
    blocking = [k for k in ("clipping", "unplanned_silence") if not checks[k]["ok"]]
    report = intent_report(intent, checks) if intent else []
    return {"version": VERSION, "blocking_failures": blocking, "checks": checks, "intent_report": report,
            "note": "only blocking_failures are technical errors; drift, LU jumps and gaps are measurements for you to judge against intent"}


LOWER_IS_BETTER = ("clipped_samples", "unplanned_silence_sec", "max_drift_ms", "max_abs_jump_lu", "low_end_overlap_sec", "true_peak_dbtp",
                   "vocal_overlap_sec", "harmonic_dissonance_mean")


def summarize(v: dict, meta: dict | None = None) -> dict:
    c = v["checks"]
    drift = [d["max_drift_ms"] for d in c["beat_alignment"]]
    jumps = [abs(j["jump_lu"]) for j in c["loudness_continuity"] if j["jump_lu"] is not None]
    rep = v.get("intent_report", [])
    return {"integrated_lufs": (meta or {}).get("integrated_lufs"), "true_peak_dbtp": c["clipping"]["true_peak_dbtp"],
            "clipped_samples": c["clipping"]["clipped_samples"],
            "unplanned_silence_sec": round(sum(r["end_sec"] - r["start_sec"] for r in c["unplanned_silence"]["regions"]), 2),
            "max_drift_ms": max(drift) if drift else None, "max_abs_jump_lu": max(jumps) if jumps else None,
            "low_end_overlap_sec": c["low_end_overlap"]["overlap_sec"], "vocal_overlap_sec": c["vocal_collision"]["seconds_overlap"],
            "harmonic_dissonance_mean": c["harmonic_clash"]["mean_dissonance"],
            "intent_pass": sum(r["status"] == "pass" for r in rep), "intent_fail": sum(r["status"] == "fail" for r in rep),
            "blocking": v["blocking_failures"]}


def compare(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if k == "blocking":
            continue
        x, y = a[k], b.get(k)
        row = {"a": x, "b": y, "delta": None if x is None or y is None else round(y - x, 3)}
        if k in LOWER_IS_BETTER and x is not None and y is not None:
            row["lower"] = "tie" if abs(y - x) < 1e-9 else ("a" if x < y else "b")
        out[k] = row
    out["blocking"] = {"a": a["blocking"], "b": b["blocking"]}
    return out
