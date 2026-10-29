from __future__ import annotations

from .constraints import Constraints
from .cost import Scorer

PAIR_RULES = ("max_bpm_jump_percent", "max_key_distance", "min_gap_between_artist_repeats", "must_not_follow", "fixed_positions", "end_with")
LIMITS = ("max_bpm_jump_percent", "max_key_distance")


def infeasible(sc: Scorer, c: Constraints, nearest: tuple) -> dict:
    total = sum(i["dur"] for i in sc.items)
    names = []
    if total < sc.target and not c.max_tracks:
        names.append("duration_min")
    if c.max_tracks and c.max_tracks * max((i["dur"] for i in sc.items), default=0) < sc.target:
        names.append("max_tracks")
    names += [r for r in PAIR_RULES if (getattr(c, r) is not None if r in LIMITS else getattr(c, r))]
    return {"infeasible": names or ["pool"], "pool_minutes": round(total / 60, 1), "target_minutes": round(sc.target / 60, 1),
            "nearest": [sc.items[i]["file"] for i in nearest], "violations_of_nearest": sc.violations(nearest) if nearest else [],
            "hint": "relax the listed constraints, enlarge the pool or shorten duration_min"}
