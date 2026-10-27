"""Rules-based planner (plan.md §3.3): ordering and per-seam TransitionPlans."""
from __future__ import annotations

from ..models import Track, TransitionPlan
from .validator import tempo_ratio, validate_plan


def camelot_distance(a: str, b: str) -> int:
    na, nb = int(a[:-1]), int(b[:-1])
    d = min((na - nb) % 12, (nb - na) % 12)
    return d + (0 if a[-1] == b[-1] else 1)


def order_tracks(tracks: list[dict]) -> list[dict]:
    """Greedy warmup -> peak ordering: start calm, then follow nearby tempo/key without big energy drops."""
    if not tracks:
        return []
    rest = sorted(tracks, key=lambda t: t["energy"])
    out = [rest.pop(0)]
    while rest:
        cur = out[-1]

        def cost(t):
            return (abs(tempo_ratio(cur["bpm"], t["bpm"]) - 1) * 100
                    + camelot_distance(cur["key"], t["key"]) * 1.5
                    + max(0.0, cur["energy"] - t["energy"]) * 4
                    + (0 if t["grid_stable"] else 10))
        nxt = min(rest, key=cost)
        rest.remove(nxt)
        out.append(nxt)
    return out


def plan_transition(a: Track, b: Track, min_from_bar: int = 0) -> TransitionPlan | None:
    """Plan an EQ-swap blend a -> b, or None when they can't be beatmatched (caller cuts instead)."""
    stretch = abs(tempo_ratio(a.bpm, b.bpm) - 1)
    if not (a.grid_stable and b.grid_stable) or stretch > 0.08:
        return _echo_out(a, b, min_from_bar)
    for L in (32, 16, 8):
        if stretch > 0.06 and L > 16:
            continue
        from_bar = ((a.n_bars - L - 1) // 8) * 8  # 8-bar boundaries from the downbeat
        if from_bar < min_from_bar or b.n_bars < L + 2:
            continue
        hold = L * 3 // 4
        plan = TransitionPlan.from_dict({
            "from_track": a.id, "to_track": b.id, "from_bar": from_bar, "to_bar": 0, "length_bars": L,
            "target_bpm": a.bpm, "tempo_ramp": {"hold_bars": hold, "ramp_bars": L - hold},
            "eq_automation": [{"bar": 0, "deck": "to", "bass": "kill"},
                              {"bar": L // 2, "deck": "to", "bass": "full"},
                              {"bar": L // 2, "deck": "from", "bass": "kill"}],
            "fx": [{"type": "lowpass_sweep", "deck": "from", "start_hz": 18000, "end_hz": 3000,
                    "bars": [L // 4, L - L // 8]}] if L >= 16 else [],
            "crossfade": {"curve": "equal_power", "bars": [0, L]},
        })
        if not validate_plan(plan, a, b):
            return plan
    return None


def _echo_out(a: Track, b: Track, min_from_bar: int) -> TransitionPlan | None:
    """No beatmatch: filter + echo the outgoing track away while the incoming one fades in on a bar line."""
    L = 4
    from_bar = ((a.n_bars - L - 1) // 4) * 4
    if from_bar < min_from_bar or b.n_bars < L + 2:
        return None
    plan = TransitionPlan.from_dict({
        "from_track": a.id, "to_track": b.id, "from_bar": from_bar, "to_bar": 0, "length_bars": L,
        "target_bpm": a.bpm, "beatmatch": False,
        "eq_automation": [{"bar": 0, "deck": "to", "bass": "kill"}, {"bar": 2, "deck": "to", "bass": "full"},
                          {"bar": 2, "deck": "from", "bass": "kill"}],
        "fx": [{"type": "lowpass_sweep", "deck": "from", "start_hz": 18000, "end_hz": 700, "bars": [0, 4]},
               {"type": "echo_out", "deck": "from", "bars": [1, 3]}],
        "crossfade": {"curve": "equal_power", "bars": [1, 3]},
    })
    return None if validate_plan(plan, a, b) else plan
