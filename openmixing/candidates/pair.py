from __future__ import annotations

import math

import numpy as np

from ..mixing.compat import camelot_relation, tempo_match
from ..mixing.keymatch import key_shift_semitones, plan_key_match
from .windows import find_windows, mean_between, rnd

MAX_OVERLAP_BARS = 32


def _extra_bars(curve, start: int, step: int, ref: float | None, limit: int, lo: int, hi: int, slack: float = 0.25) -> int:
    ref = 0.5 if ref is None else ref
    n, i = 0, start
    while n < limit and lo <= i < hi and (not curve or curve[i] <= ref + slack):
        n, i = n + 1, i + step
    return n


def extension(fa: dict, fb: dict, wa: dict, wb: dict) -> dict:
    ea = wa["energy"] if wa["energy"] is not None else None
    eb = wb["energy"] if wb["energy"] is not None else None
    xa = _extra_bars(fa["energy"], wa["start_bar"] - 1, -1, ea, MAX_OVERLAP_BARS, 0, fa["n_bars"])
    xb = _extra_bars(fb["energy"], wb["end_bar"], 1, eb, MAX_OVERLAP_BARS, 0, fb["n_bars"])
    room = int(min(wa["bars"] + xa, wb["bars"] + xb, MAX_OVERLAP_BARS))
    return {"a_extra_bars": xa, "b_extra_bars": xb, "room_with_extension": room}


def pair_report(fa: dict, fb: dict, wa: dict | None = None, wb: dict | None = None, bars: int | None = None) -> dict:
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
        vox = {"risk": rnd(vox_risk), "a_presence": rnd(np.mean(va)) if len(va) else None, "b_presence": rnd(np.mean(vb)) if len(vb) else None,
               "confidence": 0.7, "basis": "vocal stem RMS per bar over the overlap (min of both)"}
    else:
        vox = {"risk": None, "confidence": 0.0, "basis": "needs separate_stems on both tracks"}
    ea_en, eb_en = mean_between(fa["energy"], wa["start_bar"], wa["end_bar"]), mean_between(fb["energy"], wb["start_bar"], wb["end_bar"])
    return {
        "windows": {"exit": wa["id"], "entry": wb["id"], "overlap_bars_max": int(min(wa["bars"], wb["bars"], MAX_OVERLAP_BARS)),
                    "overlap_bars_measured": L, **ext},
        "tempo": {**tm, "pitch_shift_semitones_if_varispeed": rnd(pitch_st), "confidence": rnd(tconf),
                  "components": {"a_grid": fa["tempo_conf"], "b_grid": fb["tempo_conf"]}},
        "key": {"from": ea["camelot"], "to": eb["camelot"], "distance": dist, "relation": rel,
                "semitones_to_match": key_shift_semitones(ea["camelot"], eb["camelot"]), "confidence": rnd(kconf),
                "match": plan_key_match(ea, eb), "components": {"a": fa["key_conf"], "b": fb["key_conf"]}},
        "loudness": {"a_lufs": la["integrated_lufs"], "b_lufs": lb["integrated_lufs"],
                     "difference_lu": rnd(lb["integrated_lufs"] - la["integrated_lufs"], 2), "confidence": 0.9},
        "low_end": {"overlap_risk": rnd(bass_risk), "a_bass": rnd(np.mean(ba)) if len(ba) else None, "b_bass": rnd(np.mean(bb)) if len(bb) else None,
                    "confidence": 0.6,
                    "basis": "bar-level low-band (<150 Hz) energy (0..1) of the full mix of both tracks over the overlap; min() of the two",
                    "stem_low_end_carrier": {"a": ea.get("bass_carrier"), "b": eb.get("bass_carrier")}},
        "vocal": vox,
        "energy": {"a": rnd(ea_en), "b": rnd(eb_en), "delta": rnd(eb_en - ea_en) if ea_en is not None and eb_en is not None else None,
                   "confidence": 0.6},
        "phrase": {"a_exit_bar": wa["end_bar"], "a_phrase_bars": fa["phrase"],
                   "a_aligned": wa["end_bar"] % fa["phrase"] == 0 or wa["end_bar"] == fa["n_bars"],
                   "b_entry_bar": wb["start_bar"], "b_phrase_bars": fb["phrase"], "b_aligned": wb["start_bar"] % fb["phrase"] == 0,
                   "confidence": 0.7},
    }
