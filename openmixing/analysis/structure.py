from __future__ import annotations

import librosa
import numpy as np

from .confidence import attach, unit


def _norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def _label_margin(s: dict, te: float, tb: float) -> float:
    e, b = s["energy"] - te, s["bass"] - tb
    return {"drop": min(e, b), "breakdown": -b, "build": s["energy_trend"] - 0.15,
            "groove": min(b, 0.15 - s["energy_trend"]), "intro": max(-e, -b), "outro": max(-e, -b)}[s["label"]]


def structure(bf: dict, fd: float, grid_conf: float | None = None) -> dict:
    X, n, bar = bf["X"], bf["n_bars"], bf["bar_sec"]
    if n < 8:
        return {"sections": [], "phrase_bars": None, "energy_curve": [], "bass_curve": [], "onset_curve": [], "loop_candidates": [],
                "confidence": 0.0, "confidence_components": {"bars_available": round(n / 8, 3)}}
    energy, bass = _norm01(X[:, 0]), _norm01(X[:, 1])
    F = X[:, 4:80]
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-6)
    k = int(np.clip(round(n / 14), 3, 12))
    bounds = [0, *sorted(int(b) for b in librosa.segment.agglomerative(F.T, k) if 0 < b < n), n]
    raw_bounds = {int(b): int(b) for b in bounds}
    snapped = []
    for b in bounds[1:-1]:
        snapped.append(b + (1 if b % 4 == 3 else -1 if b % 4 == 1 else 0))
        raw_bounds[snapped[-1]] = int(b)
    bounds = [bounds[0], *sorted(set(snapped)), n]
    merged = [bounds[0]]
    for b in bounds[1:-1]:
        if b - merged[-1] >= 4:
            merged.append(b)
    if n - merged[-1] < 4 and len(merged) > 1:
        merged.pop()
    merged.append(n)
    secs = []
    for a, b in zip(merged, merged[1:]):
        secs.append({"start_bar": a, "end_bar": b, "bars": b - a, "raw_start_bar": raw_bounds.get(a, a), "start_sec": round(fd + a * bar, 2),
                     "end_sec": round(fd + b * bar, 2), "energy": round(float(energy[a:b].mean()), 3),
                     "bass": round(float(bass[a:b].mean()), 3),
                     "energy_trend": round(float(np.polyfit(np.arange(b - a), energy[a:b], 1)[0] * (b - a)) if b - a > 1 else 0, 3)})
    med_e, med_b = np.median([s["energy"] for s in secs]), np.median([s["bass"] for s in secs])
    for i, s in enumerate(secs):
        high, bassy = s["energy"] >= max(med_e, 0.45), s["bass"] >= max(med_b * 0.8, 0.35)
        if i == 0 and not (high and bassy):
            s["label"] = "intro"
        elif i == len(secs) - 1 and not (high and bassy):
            s["label"] = "outro"
        elif high and bassy:
            s["label"] = "drop"
        elif not bassy:
            s["label"] = "breakdown"
        elif s["energy_trend"] > 0.15:
            s["label"] = "build"
        else:
            s["label"] = "groove"
    for i in range(len(secs) - 1):
        if secs[i]["label"] in ("groove", "breakdown") and secs[i]["energy_trend"] > 0.2 and secs[i + 1]["label"] == "drop":
            secs[i]["label"] = "build"
    for i, s in enumerate(secs):
        a, b = s["start_bar"], s["end_bar"]
        mu = F[a:b].mean(0)
        contrast = None
        if i:
            pa, pb = secs[i - 1]["start_bar"], secs[i - 1]["end_bar"]
            within = np.sqrt(np.mean(np.r_[((F[pa:pb] - F[pa:pb].mean(0)) ** 2).sum(1), ((F[a:b] - mu) ** 2).sum(1)]))
            between = float(np.linalg.norm(mu - F[pa:pb].mean(0)))
            contrast = between / max(between + within, 1e-9)
        t_e, t_b = max(med_e, 0.45), max(med_b * 0.8, 0.35)
        attach(s, {"boundary_contrast": contrast, "label_margin": unit(_label_margin(s, t_e, t_b), 0, 0.25)})
    lens = [s["bars"] for s in secs[:-1] if s["bars"] >= 4] or [16]
    phrase = int(min([4, 8, 16, 32, 64], key=lambda c: abs(np.log2(c) - np.log2(np.median(lens)))))
    phrase_fit = float(np.mean([s["bars"] % phrase == 0 for s in secs[:-1]])) if len(secs) > 1 else 1.0
    Fm = X[:, 4:68] - X[:, 4:68].mean(0)
    cands = []
    for m in (1, 2, 4, 8):
        sims = []
        for k0 in range(0, n - 2 * m + 1):
            a, b = Fm[k0 : k0 + m].ravel(), Fm[k0 + m : k0 + 2 * m].ravel()
            sims.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)))
        taken = []
        for k0 in np.argsort(sims)[::-1]:
            if sims[k0] < 0.9 or len(taken) >= 2:
                break
            if all(abs(k0 - t) >= m for t in taken) and energy[k0 : k0 + m].mean() > 0.2:
                taken.append(int(k0))
                cands.append(attach({"start_bar": int(k0), "bars": m, "start_sec": round(fd + k0 * bar, 2),
                                     "repeat_similarity": round(sims[k0], 3), "energy": round(float(energy[k0 : k0 + m].mean()), 3)},
                                    {"repeat_similarity": unit(sims[k0], 0.9, 1.0), "energy": float(energy[k0 : k0 + m].mean()),
                                     "grid_confidence": grid_conf}, "product"))
    return attach({"sections": secs, "phrase_bars": phrase, "energy_curve": [round(float(v), 3) for v in energy],
                   "bass_curve": [round(float(v), 3) for v in bass], "onset_curve": [],
                   "loop_candidates": sorted(cands, key=lambda c: -c["repeat_similarity"])[:8]},
                  {"section_confidence": float(np.mean([s["confidence"] for s in secs])), "phrase_fit": phrase_fit})


def cue_points(st: dict, grid: dict, fd: float, dur: float, n_bars: int, grid_conf: float | None = None,
               downbeat_conf: float | None = None) -> dict:
    bar = 4 * grid["period"]
    secs = st["sections"]
    cues = [attach({"name": "first_downbeat", "bar": 0, "sec": round(fd, 3)}, {"downbeat": downbeat_conf, "grid": grid_conf})]
    for s in secs:
        if s["label"] in ("drop", "breakdown", "build", "outro") and not any(c["name"].startswith(s["label"]) for c in cues):
            cues.append(attach({"name": s["label"], "bar": s["start_bar"], "sec": s["start_sec"]},
                               {"section": s["confidence"], "grid": grid_conf, "downbeat": downbeat_conf}))
    phrase = st["phrase_bars"] or 16
    phrase_fit = st.get("confidence_components", {}).get("phrase_fit")
    intro = secs[0] if secs and secs[0]["label"] == "intro" else None
    mix_in = attach({"bar": 0, "sec": round(fd, 3), "why": "start of track; intro is the natural blend zone"},
                    {"downbeat": downbeat_conf, "grid": grid_conf, "section": intro["confidence"] if intro else None})
    outro = next((s for s in reversed(secs) if s["label"] == "outro"), None)
    out_bar = outro["start_bar"] if outro else max(n_bars - phrase, 0)
    out_bar = max(n_bars - (-(-(n_bars - out_bar) // 8)) * 8, 0) if n_bars - out_bar > 0 else out_bar
    mix_out = attach({"bar": int(out_bar), "sec": round(fd + out_bar * bar, 2),
                      "why": "start of outro / last phrase: start the next track here and blend over the remaining bars"},
                     {"section": outro["confidence"] if outro else 0.0, "phrase_fit": phrase_fit, "grid": grid_conf})
    return {"cue_points": cues, "mix_in": mix_in, "mix_out": mix_out,
            "intro_bars": secs[0]["bars"] if secs and secs[0]["label"] == "intro" else 0,
            "outro_bars": outro["bars"] if outro else n_bars - out_bar}
