from __future__ import annotations

from typing import Callable

from ..core.errors import OpenMixingError
from ..core.units import BAR
from ..model.clip import Clip
from .state import LOCK, at_of, clip_of, len_of, save
from .views import describe_events


def without(items: list, item_id: str, field: str) -> list:
    if item_id not in (ids := [x.id for x in items]):
        raise OpenMixingError("not_found", f"no {field[:-3]} {item_id!r}", field, f"ids: {ids}")
    return [x for x in items if x.id != item_id]


def automate(clip: str, at_bar, ramp_bars, lanes: dict[str, float], match: str, check: Callable[[Clip], None] | None = None) -> dict:
    with LOCK:
        c = clip_of(clip)
        if check:
            check(c)
        at, ramp = at_of(c, at_bar) * BAR, len_of(ramp_bars, "ramp_bars") * BAR
        for name, value in lanes.items():
            c.lanes[name].set(at, value, ramp)
        save()
    return {"clip": clip, "events": [e for e in describe_events(c) if match in e]}
