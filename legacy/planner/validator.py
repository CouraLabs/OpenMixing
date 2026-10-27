"""TransitionPlan validation (plan.md §3.2). Errors are {field, value, allowed, message}."""
from __future__ import annotations

from typing import Any

from ..models import Track, TransitionPlan

FX_TYPES = {"lowpass_sweep", "highpass_sweep", "echo_out"}
CURVES = {"equal_power", "linear"}
MAX_STRETCH = 0.08
MIN_BARS, MAX_BARS = 1, 64


def _err(field: str, value: Any, allowed: str, message: str) -> dict[str, Any]:
    return {"field": field, "value": value, "allowed": allowed, "message": message}


def tempo_ratio(target_bpm: float, native_bpm: float) -> float:
    """Stretch ratio from native to target, folding half/double-time (plan.md §3.3)."""
    return min((target_bpm / native_bpm * m for m in (0.5, 1.0, 2.0)), key=lambda r: abs(r - 1))


def validate_plan(plan: TransitionPlan, a: Track, b: Track) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    L = plan.length_bars
    if not MIN_BARS <= L <= MAX_BARS:
        errors.append(_err("length_bars", L, f"{MIN_BARS}..{MAX_BARS}", "window length out of range"))
        return errors

    for name, track in (("from_track", a), ("to_track", b)):
        if getattr(plan, name) != track.id:
            errors.append(_err(name, getattr(plan, name), track.id, "plan does not match the supplied track"))
    if plan.beatmatch and not (a.grid_stable and b.grid_stable):
        errors.append(_err("grid_stable", False, "true on both tracks",
                           "beatmatched transitions need stable grids; use echo_out/cut"))
    if not 0 <= plan.from_bar <= a.n_bars - L:
        errors.append(_err("from_bar", plan.from_bar, f"0..{a.n_bars - L}", "window runs past end of outgoing track"))
    if not 0 <= plan.to_bar <= b.n_bars - L:
        errors.append(_err("to_bar", plan.to_bar, f"0..{b.n_bars - L}", "window runs past end of incoming track"))

    if not plan.beatmatch and (plan.pitch_shift_semitones or abs(plan.target_bpm - a.bpm) > 1e-6):
        errors.append(_err("target_bpm", plan.target_bpm, f"{a.bpm} with no pitch shift",
                           "non-beatmatched transitions keep both decks at native tempo"))
    for name, track in (("from", a), ("to", b)) if plan.beatmatch else ():
        ratio = tempo_ratio(plan.target_bpm, track.bpm)
        if abs(ratio - 1) > MAX_STRETCH:
            errors.append(_err("target_bpm", plan.target_bpm, f"within ±{MAX_STRETCH:.0%} of {name} track tempo ({track.bpm})",
                               f"stretch of {ratio - 1:+.1%} on the {name} deck is too large"))
    if plan.pitch_shift_semitones not in (-1, 0, 1):
        errors.append(_err("pitch_shift_semitones", plan.pitch_shift_semitones, "-1, 0, 1", "pitch shift too large"))

    hold, ramp = plan.tempo_ramp.get("hold_bars", 0), plan.tempo_ramp.get("ramp_bars", 0)
    if hold < 0 or ramp < 0 or hold + ramp > L:
        errors.append(_err("tempo_ramp", plan.tempo_ramp, f"hold_bars + ramp_bars <= {L}", "tempo ramp exceeds window"))

    if plan.pre_loop is not None:
        n, rep = plan.pre_loop.get("bars", 0), plan.pre_loop.get("repeats", 0)
        if n not in (1, 2, 4, 8, 16) or rep < 1 or plan.from_bar - n < 0:
            errors.append(_err("pre_loop", plan.pre_loop, "bars in {1,2,4,8,16}, repeats>=1, bars<=from_bar",
                               "invalid pre_loop"))

    for i, ev in enumerate(plan.eq_automation):
        if not 0 <= ev.get("bar", -1) <= L:
            errors.append(_err(f"eq_automation[{i}].bar", ev.get("bar"), f"0..{L}", "bar outside window"))
        if ev.get("deck") not in ("from", "to"):
            errors.append(_err(f"eq_automation[{i}].deck", ev.get("deck"), "from, to", "deck must be from/to"))
        if ev.get("bass") not in ("kill", "full"):
            errors.append(_err(f"eq_automation[{i}].bass", ev.get("bass"), "kill, full", "unknown bass state"))

    for i, fx in enumerate(plan.fx):
        if fx.get("type") not in FX_TYPES:
            errors.append(_err(f"fx[{i}].type", fx.get("type"), ", ".join(sorted(FX_TYPES)), "fx type not whitelisted"))
        if fx.get("deck") not in ("from", "to"):
            errors.append(_err(f"fx[{i}].deck", fx.get("deck"), "from, to", "deck must be from/to"))
        bars = fx.get("bars")
        if not (isinstance(bars, (list, tuple)) and len(bars) == 2 and 0 <= bars[0] < bars[1] <= L):
            errors.append(_err(f"fx[{i}].bars", bars, f"[start, end] with 0 <= start < end <= {L}", "bad fx range"))
        for key in ("start_hz", "end_hz"):
            if key in fx and not 20 <= fx[key] <= 20000:
                errors.append(_err(f"fx[{i}].{key}", fx[key], "20..20000", "frequency out of range"))

    cf = plan.crossfade
    if cf.get("curve") not in CURVES:
        errors.append(_err("crossfade.curve", cf.get("curve"), ", ".join(sorted(CURVES)), "unknown curve"))
    bars = cf.get("bars")
    if bars is not None and not (len(bars) == 2 and 0 <= bars[0] <= bars[1] <= L):
        errors.append(_err("crossfade.bars", bars, f"[start, end] with 0 <= start <= end <= {L}", "bad crossfade range"))
    return errors


def phrase_warnings(plan: TransitionPlan, phrase_offset_bars: int = 0, phrase_bars: int = 16) -> list[str]:
    """Non-fatal: window should start on a phrase boundary."""
    return [f"{n} ({v}) is not on a {phrase_bars}-bar phrase boundary"
            for n, v in (("from_bar", plan.from_bar), ("to_bar", plan.to_bar))
            if (v - phrase_offset_bars) % phrase_bars]
