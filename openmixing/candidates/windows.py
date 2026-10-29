from __future__ import annotations

import numpy as np

from ..analysis.vocals import vocal_entry


def rnd(x, n=3):
    return None if x is None else round(float(x), n)


def track_facts(lib, ref: str) -> dict:
    e = lib.effective(ref)
    d = lib.analyze(e["id"])
    st, t = d["structure"], d["tempo"]
    n_bars = e["n_bars"]
    ov = e["overrides"]
    ve = vocal_entry(lib, e, n_bars)
    return {"e": e, "d": d, "n_bars": n_bars, "vocal_entry": ve, "sections": st["sections"], "phrase": st["phrase_bars"] or 16,
            "energy": st["energy_curve"], "bass": st["bass_curve"], "vocal": ve["curve"] if ve else None,
            "tempo_conf": 1.0 if "bpm" in ov else round(t["confidence"], 3),
            "key_conf": e["key_conf"]}


def mean_between(curve, a: int, b: int):
    seg = curve[max(a, 0): max(b, 0)]
    return float(np.mean(seg)) if len(seg) else None


def find_windows(f: dict, side: str) -> list[dict]:
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
            continue
        a, b = s["start_bar"], s["end_bar"]
        edge = (i == len(secs) - 1) if side == "exit" else (i == 0)
        label_conf = 0.85 if (s["label"] in ("intro", "outro") and edge) else 0.6
        raw.append((s["label"], a, b, label_conf, s["energy"], s["bass"]))
    cue = f["d"]["mix_points"]["mix_out" if side == "exit" else "mix_in"]
    if side == "exit":
        a = max(int(cue["bar"]), ph if long_track else 0)
        raw.append(("mix_out", a, n, 0.7, mean_between(f["energy"], a, n), mean_between(f["bass"], a, n)))
    else:
        b = min(n, max(ph, 8))
        raw.append(("mix_in", 0, b, 0.7, mean_between(f["energy"], 0, b), mean_between(f["bass"], 0, b)))
    out, seen = [], set()
    for label, a, b, lconf, en, bs in raw:
        if b - a < 4 or (a, b) in seen:
            continue
        seen.add((a, b))
        vox = mean_between(f["vocal"], a, b) if f["vocal"] else None
        aligned = a % ph == 0
        quiet = 1 - (0.5 * (en or 0) + 0.5 * (bs or 0))
        score = 0.5 * quiet + 0.2 * aligned + 0.2 * (label in prefer[:1]) + 0.1 * (1 - (vox or 0))
        out.append({"id": f"{side}:{label}@{a}", "side": side, "label": label, "start_bar": a, "end_bar": b, "bars": b - a,
                    "energy": rnd(en), "bass": rnd(bs), "vocal": rnd(vox), "phrase_aligned": bool(aligned),
                    "score": rnd(score), "confidence": rnd(base_conf * lconf),
                    "components": {"tempo_grid": f["tempo_conf"], "label": lconf, "phrase_bars": ph}})
    if not out:
        w = min(ph, n)
        a = max(n - w, 0) if side == "exit" else 0
        out.append({"id": f"{side}:{'tail' if side == 'exit' else 'head'}@{a}", "side": side, "label": "tail" if side == "exit" else "head",
                    "start_bar": a, "end_bar": a + w, "bars": w, "energy": None, "bass": None, "vocal": None,
                    "phrase_aligned": a % ph == 0, "score": 0.3, "confidence": rnd(base_conf * 0.4),
                    "components": {"tempo_grid": f["tempo_conf"], "label": 0.4, "phrase_bars": ph}})
    return sorted(out, key=lambda w: -w["score"])
