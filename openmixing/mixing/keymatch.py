from __future__ import annotations

from ..analysis.confidence import combine, unit
from .compat import camelot_relation

MAJOR_PC = {n: ((n - 8) * 7) % 12 for n in range(1, 13)}
MODES = ("nearest_compatible", "exact")
PREFER = ("incoming", "outgoing", "split")
LOW_CONFIDENCE = 0.4
TUNE_MIN_CENTS = 10.0
QUALITY_SHIFT = 3.0


def key_shift_semitones(cam_from: str, cam_to: str) -> int:
    d = (MAJOR_PC[int(cam_from[:-1])] - MAJOR_PC[int(cam_to[:-1])]) % 12
    return d - 12 if d > 6 else d


def shift_camelot(cam: str, semitones: int) -> str:
    return f"{(int(cam[:-1]) - 1 + 7 * semitones) % 12 + 1}{cam[-1]}"


def tuning_error(e_from: dict, e_to: dict) -> float:
    d = (e_to.get("tuning_cents", 0.0) - e_from.get("tuning_cents", 0.0) + 50.0) % 100.0 - 50.0
    return round(d, 1)


def _choose(cam_out: str, cam_in: str, mode: str, max_shift: int) -> int:
    dist = lambda s: camelot_relation(cam_out, shift_camelot(cam_in, s))[0]
    rows = sorted((dist(s), abs(s), s) for s in range(-max_shift, max_shift + 1))
    limit = 1 if mode == "nearest_compatible" else 0
    if dist(0) <= limit:
        return 0
    ok = sorted((r[1], r[2]) for r in rows if r[0] <= limit)
    if ok:
        return ok[0][1]
    return rows[0][2] if mode == "nearest_compatible" else 0


def plan_key_match(e_from: dict, e_to: dict, max_shift: int = 3, mode: str = "nearest_compatible", prefer: str = "incoming") -> dict:
    if mode not in MODES or prefer not in PREFER:
        raise ValueError(f"mode must be one of {list(MODES)} and prefer one of {list(PREFER)}")
    cam_out, cam_in = e_from["camelot"], e_to["camelot"]
    s = _choose(cam_out, cam_in, mode, int(max_shift))
    cents = tuning_error(e_from, e_to)
    fine = -cents / 100.0 if abs(cents) >= TUNE_MIN_CENTS else 0.0
    total = s + fine
    split = {"incoming": (total, 0.0), "outgoing": (0.0, -total), "split": (total / 2, -total / 2)}[prefer]
    after = shift_camelot(cam_in, s)
    dist_before, rel_before = camelot_relation(cam_out, cam_in)
    dist_after, rel_after = camelot_relation(cam_out, after)
    conf, comps = combine({"key_a": e_from.get("key_conf", 1.0), "key_b": e_to.get("key_conf", 1.0), "tuning_agreement": 1 - unit(abs(cents), 0, 50)})
    low = min(e_from.get("key_conf", 1.0), e_to.get("key_conf", 1.0)) < LOW_CONFIDENCE
    out = {"semitones_in": round(split[0], 3), "semitones_out": round(split[1], 3), "key_shift": s, "camelot_before": cam_in,
           "camelot_after": after, "distance_before": dist_before, "distance_after": dist_after, "relation_after": rel_after,
           "cents_error": cents, "confidence": conf, "confidence_components": comps, "low_confidence": low,
           "matched": dist_after == 0 or (mode == "nearest_compatible" and dist_after <= 1)}
    if max(abs(split[0]), abs(split[1])) > QUALITY_SHIFT:
        out["quality_note"] = "shifts of more than ~3 semitones with key lock audibly degrade vocals and transients"
    return out
