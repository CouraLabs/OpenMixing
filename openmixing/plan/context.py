from __future__ import annotations

import math

from ..core.errors import OpenMixingError
from ..core.units import duration_bars
from ..core.units import out_bars as units_out_bars
from .ranges import RANGES


class Ctx:
    def __init__(self, lib=None):
        self.errors: list[dict] = []
        self.lib = lib
        self.clips: dict[str, tuple[str, float]] = {}
        self.cur: str | None = None
        self.slots: dict = {}

    def add(self, code: str, field: str, message: str, fix: str | None = None) -> None:
        self.errors.append({"code": code, "field": field, "message": message, "fix": fix})

    def obj(self, v, path: str, allowed: tuple | list, required: tuple = ()) -> bool:
        if not isinstance(v, dict):
            self.add("wrong_type", path, f"expected an object, got {type(v).__name__}", "see get_plan_schema")
            return False
        for k in v:
            if k not in allowed:
                self.add("unknown_field", f"{path}.{k}", f"unsupported field {k!r}", f"allowed fields: {sorted(allowed)}")
        ok = True
        for k in required:
            if k not in v:
                self.add("missing_field", f"{path}.{k}", f"{k!r} is required", f"add {k!r}")
                ok = False
        return ok

    def num(self, v, path: str, lo: float | None = None, hi: float | None = None, key: str | None = None) -> float | None:
        if key:
            lo, hi = RANGES[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            self.add("wrong_type", path, f"expected a number, got {v!r}", "use a finite number")
            return None
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            self.add("out_of_range", path, f"{v} is outside [{lo}, {hi}]", f"use a value between {lo} and {hi}")
            return None
        return float(v)

    def length(self, v, path: str, lo: float | None = None, hi: float | None = None, key: str | None = None) -> float | None:
        if isinstance(v, str):
            try:
                v = duration_bars(v, path)
            except OpenMixingError as e:
                self.add(e.code, path, e.message, e.fix)
                return None
        return self.num(v, path, lo, hi, key)

    def out_bars(self, v, path: str, clip_id: str | None = None) -> float | None:
        if isinstance(v, str):
            ref = self.clips.get(clip_id or self.cur)
            to_src = (lambda x: self.lib.to_bars(ref[0], x, path)) if ref and self.lib else None
            try:
                v = units_out_bars(v, ref[1] if ref else 0.0, to_src, path)
            except OpenMixingError as e:
                self.add(e.code, path, e.message, e.fix)
                return None
            except (ValueError, KeyError) as e:
                self.add("invalid_position", path, f"cannot read {v!r} as a position: {e.args[0] if e.args else e}",
                         "use bars, 'N beats', 'cue:<name>', 'section:<label>' or '95.5s'")
                return None
        return self.num(v, path, 0.0, None)
