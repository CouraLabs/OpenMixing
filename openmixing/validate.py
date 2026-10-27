"""Measure a rendered set against its plan: beat drift, loudness continuity, unplanned silence, clipping."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .models import SR
from .setmodel import SetModel, TempoMap, resolve_starts

SILENCE_DB = -60.0
MIN_SILENCE_S = 0.4
VERSION = 2
LOW_END_DB = -35.0     # a deck's low band (<~200 Hz) above this counts as carrying bass
DRIFT_OK_MS = 25.0
INTENT_KEYS = ("beat_drift_ms_max", "beat_drift_intended", "loudness_jump_lu", "silence_intended", "low_end_overlap_max_sec",
               "low_end_clash_intended", "true_peak_max_dbtp")


def _beat_times(c, tm: TempoMap, start: float) -> np.ndarray:
    return start + np.asarray(tm.beat_to_sec(np.arange(0, np.floor(c.length_beats) + 1, 1.0)))


def beat_drift(model: SetModel, info) -> list[dict]:
    """Per overlapping clip pair: offset (ms) of each beat of one clip from the nearest beat of the other, over the overlap."""
    starts = resolve_starts(model, lambda t: float(info(t)["bpm"]))
    maps = {c.id: TempoMap(c, float(info(c.track)["bpm"])) for c in model.clips}
    beats = {c.id: _beat_times(c, maps[c.id], starts[c.id]) for c in model.clips}
    out = []
    for i, a in enumerate(model.clips):
        for b in model.clips[i + 1:]:
            lo, hi = max(beats[a.id][0], beats[b.id][0]), min(beats[a.id][-1], beats[b.id][-1])
            if hi - lo < 1.0:
                continue
            ta = beats[a.id][(beats[a.id] >= lo) & (beats[a.id] <= hi)]
            tb = beats[b.id]
            d = np.array([np.min(np.abs(tb - t)) for t in ta]) * 1000.0
            out.append({"clips": [a.id, b.id], "overlap_sec": [round(float(lo), 2), round(float(hi), 2)],
                        "max_drift_ms": round(float(d.max()), 1), "mean_drift_ms": round(float(d.mean()), 1),
                        "ok": bool(d.max() < DRIFT_OK_MS)})
    return out


def _lufs(x: np.ndarray) -> float | None:
    if x.shape[1] < int(0.5 * SR):
        return None
    v = pyln.Meter(SR).integrated_loudness(x.T)
    return None if not np.isfinite(v) else round(float(v), 2)


def loudness_continuity(x: np.ndarray, t0: float, rows: list[dict], window: float = 8.0) -> list[dict]:
    """LU change across each overlap: loudness of `window` seconds before the overlap vs after it."""
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            lo, hi = max(a["start_sec"], b["start_sec"]), min(a["end_sec"], b["end_sec"])
            if hi - lo < 1.0:
                continue
            seg = lambda s, e: x[:, max(int((s - t0) * SR), 0): max(int((e - t0) * SR), 0)]
            before, after = _lufs(seg(lo - window, lo)), _lufs(seg(hi, hi + window))
            out.append({"clips": [a["clip"], b["clip"]], "before_lufs": before, "after_lufs": after,
                        "jump_lu": None if before is None or after is None else round(after - before, 2)})
    return out


def silences(x: np.ndarray, t0: float, rows: list[dict]) -> list[dict]:
    """Silent stretches; `planned` is True when no clip covers them (a gap the plan left)."""
    blk = SR // 20
    n = x.shape[1] // blk
    if n == 0:
        return []
    rms = np.sqrt((x[:, : n * blk].reshape(2, n, blk) ** 2).mean(axis=(0, 2)) + 1e-12)
    quiet = 20 * np.log10(rms) < SILENCE_DB
    out, i = [], 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        a, b = t0 + i * blk / SR, t0 + j * blk / SR
        if b - a >= MIN_SILENCE_S:
            covered = any(r["start_sec"] < b and r["end_sec"] > a for r in rows)
            out.append({"start_sec": round(a, 2), "end_sec": round(b, 2), "planned": not covered})
        i = j
    return out


def low_end_overlap(low_db: dict, block_sec: float, t0: float) -> dict:
    """Time during which two or more decks carry bass at once (per-deck low band, measured on the render)."""
    if len(low_db) < 2:
        return {"overlap_sec": 0.0, "regions": [], "threshold_db": LOW_END_DB}
    ids = list(low_db)
    n = min(len(v) for v in low_db.values())
    on = np.array([np.asarray(low_db[i])[:n] > LOW_END_DB for i in ids])
    both = on.sum(axis=0) >= 2
    regions, i = [], 0
    while i < n:
        if not both[i]:
            i += 1
            continue
        j = i
        while j < n and both[j]:
            j += 1
        regions.append({"start_sec": round(t0 + i * block_sec, 2), "end_sec": round(t0 + j * block_sec, 2),
                        "clips": [c for k, c in enumerate(ids) if on[k, i:j].any()]})
        i = j
    return {"overlap_sec": round(float(both.sum() * block_sec), 2), "regions": regions[:8], "threshold_db": LOW_END_DB}


def intent_report(intent: dict, checks: dict) -> list[dict]:
    """Pass/fail for every item the plan declared. A declared clash passes when it happened; it fails when it did not."""
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
    return out


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
    checks = {
        "clipping": {"clipped_samples": clipped, "true_peak_dbtp": meta.get("true_peak_dbtp"),
                     "ok": clipped == 0 and (meta.get("true_peak_dbtp") or -99) <= -0.1},
        "unplanned_silence": {"regions": [s for s in sil if not s["planned"]], "ok": not any(not s["planned"] for s in sil)},
        "planned_gaps": [s for s in sil if s["planned"]],
        "low_end_overlap": low_end_overlap(meta.get("low_db") or {}, meta.get("low_block_sec", 0.25), t0),
        "beat_alignment": drift,
        "loudness_continuity": loudness_continuity(x, t0, rows),
    }
    blocking = [k for k in ("clipping", "unplanned_silence") if not checks[k]["ok"]]
    report = intent_report(intent, checks) if intent else []
    return {"version": VERSION, "blocking_failures": blocking, "checks": checks, "intent_report": report,
            "note": "only blocking_failures are technical errors; drift, LU jumps and gaps are measurements for you to judge against intent"}


LOWER_IS_BETTER = ("clipped_samples", "unplanned_silence_sec", "max_drift_ms", "max_abs_jump_lu", "low_end_overlap_sec", "true_peak_dbtp")


def summarize(v: dict, meta: dict | None = None) -> dict:
    """A flat set of numbers from a validation block, the unit of comparison between renders."""
    c = v["checks"]
    drift = [d["max_drift_ms"] for d in c["beat_alignment"]]
    jumps = [abs(j["jump_lu"]) for j in c["loudness_continuity"] if j["jump_lu"] is not None]
    rep = v.get("intent_report", [])
    return {"integrated_lufs": (meta or {}).get("integrated_lufs"), "true_peak_dbtp": c["clipping"]["true_peak_dbtp"],
            "clipped_samples": c["clipping"]["clipped_samples"],
            "unplanned_silence_sec": round(sum(r["end_sec"] - r["start_sec"] for r in c["unplanned_silence"]["regions"]), 2),
            "max_drift_ms": max(drift) if drift else None, "max_abs_jump_lu": max(jumps) if jumps else None,
            "low_end_overlap_sec": c["low_end_overlap"]["overlap_sec"],
            "intent_pass": sum(r["status"] == "pass" for r in rep), "intent_fail": sum(r["status"] == "fail" for r in rep),
            "blocking": v["blocking_failures"]}


def compare(a: dict, b: dict) -> dict:
    """Metric by metric difference b - a. `lower` names the render with the smaller value of a lower-is-better metric; it is not a verdict."""
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
