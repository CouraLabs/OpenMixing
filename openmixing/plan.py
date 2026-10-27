"""Declarative set plan: a versioned JSON document built only from allowlisted operations.

`build_plan` is the single code path: it checks every field (collecting all errors, each with a JSON-path `field`) and
builds a SetModel. `validate_plan` returns the errors; `apply_plan` raises PlanError instead of returning a half-built set.
Nothing in a plan is ever evaluated as code.
"""
from __future__ import annotations

import math
import re

from .dsp import fx as fxm
from .errors import OpenMixingError
from .models import BEAT_DIVISIONS, BEAT_LABELS, parse_beats
from .setmodel import LANES, LEGACY_STEM_NAMES, STEM_LANES, Clip, FxEvent, Lane, LoopEvent, SetModel, resolve_starts
from .validate import INTENT_KEYS
from .units import duration_bars, out_bars as units_out_bars, to_bars

PLAN_VERSION = "1.0"
BAR = 4.0
CURVES = ("linear", "equal_power", "log", "s_curve", "power", "sharp_cut")
BANDS = ("low", "mid", "high")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
KILL_DB = -80.0
RANGES = {"eq": (-80.0, 6.0), "filter": (-1.0, 1.0), "volume": (0.0, 1.5), "stem": (0.0, 1.5), "bpm": (40.0, 250.0),
          "gain_db": (-40.0, 12.0), "semitones": (-12.0, 12.0), "ramp_bars": (0.0, 256.0), "wet": (0.0, 1.0),
          "exponent": (0.1, 10.0), "offset_ms": (-2000.0, 2000.0)}


class PlanError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__(f"plan has {len(errors)} error(s): " + "; ".join(f"{e['field']}: {e['message']}" for e in errors[:5]))
        self.errors = errors


# ---------------------------------------------------------------- schema
def _event_schema(value_key: str, value: dict, desc: str) -> dict:
    return {"description": desc, "oneOf": [
        {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["at", value_key], "properties": {
            "at": {"$ref": "#/$defs/out_position"}, value_key: value,
            "ramp_bars": {"type": ["number", "string"], "minimum": 0, "maximum": 256, "default": 0, "description": "bars or \"N beats\"; 0 snaps (de-clicked ~10 ms)"},
            "curve": {"enum": list(CURVES), "default": "linear"},
            "exponent": {"type": "number", "minimum": 0.1, "maximum": 10, "description": "k of the u^k shape, curve=power only"}}}},
        {"type": "object", "additionalProperties": False, "required": ["points"], "properties": {
            "points": {"type": "array", "items": {"type": "array", "prefixItems": [{"$ref": "#/$defs/out_position"}, value],
                                                  "minItems": 2, "maxItems": 2}}},
         "description": "raw envelope: [at, value] pairs, linear between points"}]}


def _num_range(key: str) -> dict:
    lo, hi = RANGES[key]
    return {"type": "number", "minimum": lo, "maximum": hi}


def plan_schema() -> dict:
    eq_val = {"oneOf": [_num_range("eq"), {"const": "kill"}], "description": "dB, -80 = kill, 0 = flat"}
    fx_names = list(fxm.CATALOG)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"openmixing://plan-schema/{PLAN_VERSION}",
        "title": "OpenMixing set plan", "version": PLAN_VERSION, "type": "object", "additionalProperties": False,
        "required": ["version", "clips"],
        "properties": {
            "version": {"const": PLAN_VERSION}, "name": {"type": "string"}, "notes": {"type": "string"}, "intent": {"$ref": "#/$defs/intent"},
            "clips": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/clip"}}},
        "$defs": {
            "intent": {"type": "object", "additionalProperties": False, "description":
                       "What you expect the render to show. The render's validation.intent_report marks each declared item pass or fail; "
                       "a declared clash passes when it happened.", "properties": {
                           "beat_drift_ms_max": {"type": "number", "minimum": 0}, "beat_drift_intended": {"type": "boolean"},
                           "loudness_jump_lu": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                                                "description": "[min, max] LU change across every overlap"},
                           "silence_intended": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                                "description": "[start_sec, end_sec] on the set timeline"},
                           "low_end_overlap_max_sec": {"type": "number", "minimum": 0}, "low_end_clash_intended": {"type": "boolean"},
                           "true_peak_max_dbtp": {"type": "number"}}},
            "out_position": {"description": "Position inside this clip's playback: output bars (bar 0 = clip start), \"N beats\", or a place in the source "
                                            "track (\"cue:<name>\", \"section:<label>+N\", \"95.5s\"), converted with the clip's src_start.",
                             "oneOf": [{"type": "number", "minimum": 0}, {"type": "string"}]},
            "src_position": {"description": "Position in the source track, in bars (0 = first downbeat), or \"64 beats\", \"95.5s\", "
                                            "\"cue:<name>\", \"section:<label>+N\".",
                             "oneOf": [{"type": "number", "minimum": 0}, {"type": "string"}]},
            "clip": {
                "type": "object", "additionalProperties": False, "required": ["track"], "properties": {
                    "id": {"type": "string", "pattern": ID_RE.pattern, "description": "your name for the clip; default c1, c2 ..."},
                    "track": {"type": "string", "description": "track id, file name or unique part of the name"},
                    "label": {"type": "string"},
                    "src_start": {"$ref": "#/$defs/src_position", "default": 0},
                    "bars": {"$ref": "#/$defs/src_position", "description": "length played; default to the end of the track"},
                    "start": {"description": "default: right after the previous clip in this list", "oneOf": [
                        {"type": "object", "additionalProperties": False, "required": ["at_sec"], "properties": {
                            "at_sec": {"type": "number", "minimum": 0}, "offset_ms": _num_range("offset_ms")}},
                        {"type": "object", "additionalProperties": False, "required": ["over", "at_bar"], "properties": {
                            "over": {"type": "string", "description": "id of another clip in this plan"},
                            "at_bar": {"$ref": "#/$defs/out_position"}, "offset_ms": _num_range("offset_ms")}}]},
                    "bpm": {**_num_range("bpm"), "description": "constant playback tempo (default: native)"},
                    "tempo": _event_schema("bpm", _num_range("bpm"), "tempo moves (pitch-fader), needs no bpm"),
                    "gain_db": _num_range("gain_db"), "semitones": _num_range("semitones"),
                    "key_lock": {"type": "boolean", "default": True}, "stems": {"type": "boolean", "default": False},
                    "eq": {"type": "object", "additionalProperties": False,
                           "properties": {b: _event_schema("gain_db", eq_val, f"{b} band") for b in BANDS}},
                    "filter": _event_schema("value", _num_range("filter"), "-1 low-pass closed, 0 off, +1 high-pass closed"),
                    "volume": _event_schema("level", _num_range("volume"), "channel fader, 1 = unity"),
                    "stem_levels": {"type": "object", "additionalProperties": False, "description": "needs stems=true and separate_stems",
                                    "properties": {s: _event_schema("level", _num_range("stem"), f"{s} stem fader") for s in STEM_LANES}},
                    "fx": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["fx", "at", "bars"],
                           "properties": {"id": {"type": "string"}, "fx": {"enum": fx_names}, "at": {"$ref": "#/$defs/out_position"},
                                          "bars": {"type": ["number", "string"], "exclusiveMinimum": 0, "description": "bars or \"N beats\""},
                                          "beats": {"enum": [BEAT_LABELS[b] for b in BEAT_DIVISIONS], "default": "1"},
                                          "wet": {**_num_range("wet"), "default": 1.0},
                                          "stem": {"enum": list(STEM_LANES), "description": "apply to this stem only (needs stems=true)"},
                                          "params": {"type": "object", "description": "per-effect numbers, see list_fx"}}}},
                    "loops": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["at", "size_beats"],
                              "properties": {"id": {"type": "string"}, "at": {"$ref": "#/$defs/out_position"},
                                             "size_beats": {"enum": [BEAT_LABELS[b] for b in BEAT_DIVISIONS]},
                                             "bars": {"type": ["number", "string"], "exclusiveMinimum": 0}, "repeats": {"type": "integer", "minimum": 1},
                                             "roll": {"type": "boolean", "default": False}},
                              "description": "give exactly one of bars / repeats; loops may not overlap"}}}}},
        "curves": {"linear": "straight ramp", "equal_power": "sin/cos fade (constant power when one rises while the other falls)",
                   "log": "log10(1+9u): fast start, slow end (mirrored when falling)", "s_curve": "cosine ease in/out",
                   "power": "u^exponent: slow start, surge at the end when exponent > 1 (tension builds)",
                   "sharp_cut": "hold, then jump at the end of the ramp"},
    }


# ---------------------------------------------------------------- helpers
class _Ctx:
    def __init__(self, lib=None):
        self.errors: list[dict] = []
        self.lib = lib
        self.clips: dict[str, tuple[str, float]] = {}  # clip id -> (track id, src_start bar), filled as clips are built
        self.cur: str | None = None

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
        """A length in bars: a number or "N beats"."""
        if isinstance(v, str):
            try:
                v = duration_bars(v, path)
            except OpenMixingError as e:
                self.add(e.code, path, e.message, e.fix)
                return None
        return self.num(v, path, lo, hi, key)

    def out_bars(self, v, path: str, clip_id: str | None = None) -> float | None:
        """Position inside a clip (the current one unless `clip_id`) in output bars: number, "N beats", or a source-track
        position ("cue:<name>", "section:<label>+N", "95.5s") converted with that clip's src_start."""
        if isinstance(v, str):
            ref = self.clips.get(clip_id or self.cur)
            to_src = (lambda x: self.lib.to_bars(ref[0], x, path)) if ref and self.lib else None
            try:
                v = units_out_bars(v, ref[1] if ref else 0.0, to_src, path)
            except OpenMixingError as e:
                self.add(e.code, path, e.message, e.fix)
                return None
            except (ValueError, KeyError) as e:
                self.add("invalid_position", path, f"cannot read {v!r} as a position: {e.args[0] if e.args else e}", "use bars, 'N beats', 'cue:<name>', 'section:<label>' or '95.5s'")
                return None
        return self.num(v, path, 0.0, None)


def _shape(curve: str, k: float, rising: bool):
    def base(p: float) -> float:
        if curve == "equal_power":
            return math.sin(p * math.pi / 2)
        if curve == "log":
            return math.log10(1 + 9 * p)
        if curve == "s_curve":
            return 0.5 - 0.5 * math.cos(math.pi * p)
        if curve == "power":
            return p ** k
        return p
    if curve in ("equal_power", "log") and not rising:
        return lambda p: 1 - base(1 - p)
    return base


def _write(lane: Lane, at: float, value: float, ramp: float, curve: str, k: float) -> None:
    if curve == "linear" or ramp <= 0:
        lane.set(at, value, ramp)
    elif curve == "sharp_cut":
        lane.set(at + ramp, value, 0.0)
    else:
        v0 = lane.value_at(at)
        sh = _shape(curve, k, value >= v0)
        lane.curve(at, ramp, lambda p: v0 + (value - v0) * sh(p), steps_per_beat=4)


def _envelope(cx: _Ctx, v, path: str, lane: Lane, value_key: str, lo: float, hi: float, length: float, kill: bool = False) -> None:
    if isinstance(v, dict):
        if not cx.obj(v, path, ("points",), ("points",)):
            return
        pts = v["points"]
        if not isinstance(pts, list):
            cx.add("wrong_type", f"{path}.points", "expected a list of [at, value] pairs")
            return
        out = []
        for j, p in enumerate(pts):
            pp = f"{path}.points[{j}]"
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                cx.add("wrong_type", pp, "expected [at, value]")
                continue
            a, val = cx.out_bars(p[0], f"{pp}[0]"), cx.num(p[1], f"{pp}[1]", lo, hi)
            if a is not None and a > length + 1e-9:
                cx.add("out_of_range", f"{pp}[0]", f"{a:g} is beyond the clip ({length:g} bars)", "shorten the envelope or lengthen the clip")
            elif a is not None and val is not None:
                out.append([a * BAR, val])
        if sorted(p[0] for p in out) != [p[0] for p in out]:
            cx.add("unsorted_points", f"{path}.points", "points must be in increasing time order", "sort by the first element")
        lane.points = out
        return
    if not isinstance(v, list):
        cx.add("wrong_type", path, "expected a list of events or {\"points\": [...]}", "see get_plan_schema")
        return
    evs = []
    for j, e in enumerate(v):
        ep = f"{path}[{j}]"
        if not cx.obj(e, ep, ("at", value_key, "ramp_bars", "curve", "exponent"), ("at", value_key)):
            continue
        at = cx.out_bars(e["at"], f"{ep}.at")
        if at is not None and at > length + 1e-9:
            cx.add("out_of_range", f"{ep}.at", f"{at:g} is beyond the clip ({length:g} bars)", "move it inside the clip")
            at = None
        raw = e[value_key]
        if kill and raw == "kill":
            val = KILL_DB
        else:
            val = cx.num(raw, f"{ep}.{value_key}", lo, hi)
        ramp = cx.length(e.get("ramp_bars", 0.0), f"{ep}.ramp_bars", key="ramp_bars")
        curve = e.get("curve", "linear")
        if curve not in CURVES:
            cx.add("unknown_curve", f"{ep}.curve", f"unknown curve {curve!r}", f"one of {list(CURVES)}")
            curve = None
        k = cx.num(e.get("exponent", 2.0), f"{ep}.exponent", key="exponent")
        if "exponent" in e and curve not in (None, "power"):
            cx.add("unused_field", f"{ep}.exponent", "exponent only applies to curve 'power'", "remove it or use curve 'power'")
        if None not in (at, val, ramp, curve, k):
            evs.append((at, val, ramp, curve, k))
    for at, val, ramp, curve, k in sorted(evs, key=lambda e: e[0]):
        _write(lane, at * BAR, val, ramp * BAR, curve, k)


def _src(cx: _Ctx, lib, tid: str, v, path: str) -> float | None:
    try:
        return lib.to_bars(tid, v, path)
    except OpenMixingError as e:
        cx.add(e.code, path, e.message, e.fix)
    except (ValueError, KeyError) as e:
        cx.add("invalid_position", path, str(e.args[0] if e.args else e), "use bars, 'N beats', 'cue:<name>' or 'section:<label>'")
    return None


CLIP_KEYS = ("id", "track", "label", "src_start", "bars", "start", "bpm", "tempo", "gain_db", "semitones", "key_lock", "stems",
             "eq", "filter", "volume", "stem_levels", "fx", "loops")


def _build_clip(cx: _Ctx, lib, c, i: int, ids: set[str], seq: int) -> Clip | None:
    p = f"clips[{i}]"
    if not cx.obj(c, p, CLIP_KEYS, ("track",)):
        return None
    cid = c.get("id", f"c{seq}")
    if not isinstance(cid, str) or not ID_RE.match(cid):
        cx.add("invalid_id", f"{p}.id", f"bad clip id {cid!r}", "use letters, digits, _ or - (max 40)")
        cid = f"c{seq}"
    if cid in ids:
        cx.add("duplicate_id", f"{p}.id", f"clip id {cid!r} is used twice", "give each clip a unique id")
    ids.add(cid)
    try:
        e = lib.effective(c["track"])
        d = lib.analyze(e["id"])
    except (KeyError, ValueError) as ex:
        cx.add("track_not_found", f"{p}.track", str(ex.args[0] if ex.args else ex), "call list_tracks and use an id or file name")
        return None
    src = _src(cx, lib, e["id"], c.get("src_start", 0.0), f"{p}.src_start")
    n_bars = d["tempo"]["n_bars"] if "bpm" not in e["overrides"] else int((e["duration"] - e["first_downbeat"]) * e["bpm"] / 240)
    length = None
    if "bars" in c:
        length = _src(cx, lib, e["id"], c["bars"], f"{p}.bars")
    elif src is not None:
        length = n_bars - src
    if src is not None and src >= n_bars:
        cx.add("out_of_range", f"{p}.src_start", f"{src:g} is past the end of the track ({n_bars} bars)", f"use a bar below {n_bars}")
        src = None
    if length is not None and length <= 0:
        cx.add("out_of_range", f"{p}.bars", "the clip must play for more than 0 bars", "give a positive length")
        length = None
    clip = Clip(cid, e["id"], (src or 0.0) * BAR, (length or 1.0) * BAR, label=str(c.get("label") or e["file"].rsplit(".", 1)[0]))
    L = length if length is not None else 1.0
    cx.clips[cid], cx.cur = (e["id"], (src or 0.0)), cid
    for key, attr in (("gain_db", "gain_db"), ("semitones", "semitones")):
        if key in c and (v := cx.num(c[key], f"{p}.{key}", key=key)) is not None:
            setattr(clip, attr, v)
    if "key_lock" in c:
        if isinstance(c["key_lock"], bool):
            clip.key_lock = c["key_lock"]
        else:
            cx.add("wrong_type", f"{p}.key_lock", "expected true or false")
    if "stems" in c:
        if not isinstance(c["stems"], bool):
            cx.add("wrong_type", f"{p}.stems", "expected true or false")
        elif c["stems"] and not e.get("stems"):
            cx.add("missing_stems", f"{p}.stems", f"{e['file']} has no stems yet" if not e.get("stems_legacy")
                   else f"{e['file']} only has the old 3-stem layout", "call separate_stems" + (" with force=True" if e.get("stems_legacy") else " first"))
        else:
            clip.stems = c["stems"]
    if "bpm" in c and (v := cx.num(c["bpm"], f"{p}.bpm", key="bpm")) is not None:
        clip.tempo.points = [[0.0, v]]
    if "tempo" in c:
        if not clip.tempo.points:
            clip.tempo.points = [[0.0, float(e["bpm"])]]
        _envelope(cx, c["tempo"], f"{p}.tempo", clip.tempo, "bpm", *RANGES["bpm"], L)
    if "eq" in c and cx.obj(c["eq"], f"{p}.eq", BANDS):
        for b in BANDS:
            if b in c["eq"]:
                _envelope(cx, c["eq"][b], f"{p}.eq.{b}", clip.lanes[f"eq_{b}"], "gain_db", *RANGES["eq"], L, kill=True)
    for key, vk, rk in (("filter", "value", "filter"), ("volume", "level", "volume")):
        if key in c:
            _envelope(cx, c[key], f"{p}.{key}", clip.lanes[key], vk, *RANGES[rk], L)
    if isinstance(c.get("stem_levels"), dict):  # plans written for the old 3-stem layout
        sl = dict(c["stem_levels"])
        for old, new in LEGACY_STEM_NAMES.items():
            if old in sl:
                for n in new:
                    sl.setdefault(n, sl[old])
                del sl[old]
        c = {**c, "stem_levels": sl}
    if "stem_levels" in c and cx.obj(c["stem_levels"], f"{p}.stem_levels", tuple(STEM_LANES)):
        if not clip.stems and c["stem_levels"]:
            cx.add("stems_off", f"{p}.stem_levels", "stem levels need stems=true on this clip", "set \"stems\": true (after separate_stems)")
        for s, lane in STEM_LANES.items():
            if s in c["stem_levels"]:
                _envelope(cx, c["stem_levels"][s], f"{p}.stem_levels.{s}", clip.lanes[lane], "level", *RANGES["stem"], L)
    _fx(cx, c.get("fx", []), p, clip, L)
    _loops(cx, c.get("loops", []), p, clip, L)
    return clip


def _fx(cx: _Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.fx", "expected a list")
        return
    for j, f in enumerate(items):
        fp = f"{p}.fx[{j}]"
        if not cx.obj(f, fp, ("id", "fx", "at", "bars", "beats", "wet", "params", "stem"), ("fx", "at", "bars")):
            continue
        name = f["fx"]
        spec = fxm.CATALOG.get(name) if isinstance(name, str) else None
        if spec is None:
            cx.add("unknown_fx", f"{fp}.fx", f"unknown effect {name!r}", f"one of {list(fxm.CATALOG)}")
        stem = f.get("stem")
        if stem is not None:
            if stem not in STEM_LANES:
                cx.add("unknown_stem", f"{fp}.stem", f"unknown stem {stem!r}", f"one of {list(STEM_LANES)}")
                stem = None
                spec = None
            elif not clip.stems:
                cx.add("stems_off", f"{fp}.stem", "a stem effect needs stems=true on this clip", "set \"stems\": true (after separate_stems) or drop \"stem\"")
                spec = None
        at, dur = cx.out_bars(f["at"], f"{fp}.at"), cx.length(f["bars"], f"{fp}.bars", 1e-6, None)
        if at is not None and at > L + 1e-9:
            cx.add("out_of_range", f"{fp}.at", f"{at:g} is beyond the clip ({L:g} bars)", "move it inside the clip")
        try:
            beats = parse_beats(f.get("beats", 1.0))
        except (ValueError, TypeError):
            beats = None
            cx.add("bad_beats", f"{fp}.beats", f"beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {f.get('beats')!r}",
                   "use a tempo-sync division")
        wet = cx.num(f.get("wet", 1.0), f"{fp}.wet", key="wet")
        params = f.get("params", {})
        if not isinstance(params, dict):
            cx.add("wrong_type", f"{fp}.params", "expected an object")
            params = {}
        for k, v in params.items():
            if spec is not None and k not in spec.params:
                cx.add("unknown_param", f"{fp}.params.{k}", f"{name} has no parameter {k!r}", f"allowed: {sorted(spec.params)}")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                cx.add("wrong_type", f"{fp}.params.{k}", "effect parameters must be finite numbers")
        if spec and None not in (at, dur, beats, wet):
            clip.fx.append(FxEvent(str(f.get("id") or f"fx{len(clip.fx) + 1}"), name, at * BAR, dur * BAR, beats, wet, dict(params), stem))


def _loops(cx: _Ctx, items, p: str, clip: Clip, L: float) -> None:
    if not isinstance(items, list):
        cx.add("wrong_type", f"{p}.loops", "expected a list")
        return
    for j, l in enumerate(items):
        lp = f"{p}.loops[{j}]"
        if not cx.obj(l, lp, ("id", "at", "size_beats", "bars", "repeats", "roll"), ("at", "size_beats")):
            continue
        at = cx.out_bars(l["at"], f"{lp}.at")
        try:
            size = parse_beats(l["size_beats"])
        except (ValueError, TypeError):
            size = None
            cx.add("bad_beats", f"{lp}.size_beats", f"size_beats must be one of {[BEAT_LABELS[b] for b in BEAT_DIVISIONS]}, got {l['size_beats']!r}",
                   "use a loop size from the list")
        if ("bars" in l) == ("repeats" in l):
            cx.add("bars_or_repeats", lp, "give exactly one of bars / repeats", "remove one of them")
            continue
        if "bars" in l:
            n = cx.length(l["bars"], f"{lp}.bars", 1e-6, None)
            length = None if n is None else n * BAR
        else:
            r = l["repeats"]
            if isinstance(r, bool) or not isinstance(r, int) or r < 1:
                cx.add("out_of_range", f"{lp}.repeats", f"repeats must be an integer >= 1, got {r!r}", "use a whole number")
                length = None
            else:
                length = None if size is None else size * r
        roll = l.get("roll", False)
        if not isinstance(roll, bool):
            cx.add("wrong_type", f"{lp}.roll", "expected true or false")
            roll = False
        if at is not None and at > L + 1e-9:
            cx.add("out_of_range", f"{lp}.at", f"{at:g} is beyond the clip ({L:g} bars)", "move it inside the clip")
        elif None not in (at, size, length):
            clip.loops.append(LoopEvent(str(l.get("id") or f"lp{len(clip.loops) + 1}"), at * BAR, size, length, roll))
    try:
        clip.validate_loops()
    except ValueError as e:
        cx.add("loop_overlap", f"{p}.loops", str(e), "loops may not overlap: shorten one or move it")


def _start(cx: _Ctx, c, i: int, clip: Clip, prev: Clip | None, ids_all: set[str]) -> None:
    p = f"clips[{i}].start"
    s = c.get("start")
    if s is None:
        clip.start = {"anchor": prev.id, "anchor_beat": prev.length_beats} if prev else {"at_sec": 0.0}
        return
    if not isinstance(s, dict):
        cx.add("wrong_type", p, "expected an object")
        return
    if "over" in s:
        if not cx.obj(s, p, ("over", "at_bar", "offset_ms"), ("over", "at_bar")):
            return
        if s["over"] not in ids_all:
            cx.add("unknown_clip", f"{p}.over", f"no clip {s['over']!r} in this plan", f"clip ids: {sorted(ids_all)}")
            return
        if s["over"] == clip.id:
            cx.add("anchor_cycle", f"{p}.over", "a clip cannot be anchored to itself", "anchor it to another clip")
            return
        at = cx.out_bars(s["at_bar"], f"{p}.at_bar", s["over"])
        if at is not None:
            clip.start = {"anchor": s["over"], "anchor_beat": at * BAR}
    else:
        if not cx.obj(s, p, ("at_sec", "offset_ms"), ("at_sec",)):
            return
        at = cx.num(s["at_sec"], f"{p}.at_sec", 0.0, None)
        if at is not None:
            clip.start = {"at_sec": at}
    if "offset_ms" in s and (o := cx.num(s["offset_ms"], f"{p}.offset_ms", key="offset_ms")) is not None:
        clip.start["offset_ms"] = o


def build_plan(plan, lib) -> tuple[SetModel | None, list[dict]]:
    """Check `plan` and build the SetModel. Returns (model, []) or (None, errors)."""
    cx = _Ctx(lib)
    if not cx.obj(plan, "plan", ("version", "name", "notes", "clips", "intent"), ("version", "clips")):
        return None, cx.errors
    if plan["version"] != PLAN_VERSION:
        cx.add("unsupported_version", "plan.version", f"plan version {plan['version']!r} is not supported", f"use {PLAN_VERSION!r}")
    if not isinstance(plan["clips"], list) or not plan["clips"]:
        cx.add("wrong_type", "plan.clips", "expected a non-empty list of clips")
        return None, cx.errors
    ids_all = {c["id"] for c in plan["clips"] if isinstance(c, dict) and isinstance(c.get("id"), str)}
    ids_all |= {f"c{n}" for n, c in enumerate(plan["clips"], 1) if isinstance(c, dict) and "id" not in c}
    ids: set[str] = set()
    clips: list[Clip] = []
    prev = None
    for i, c in enumerate(plan["clips"]):
        clip = _build_clip(cx, lib, c, i, ids, i + 1)
        if clip is None:
            prev = None
            continue
        _start(cx, c, i, clip, prev, ids_all)
        clips.append(clip)
        prev = clip
    name = plan.get("name", "plan")
    if not isinstance(name, str) or not name:
        cx.add("wrong_type", "plan.name", "expected a non-empty string")
        name = "plan"
    model = SetModel(name, clips, str(plan.get("notes", "")), intent=_intent(cx, plan.get("intent")))
    if not cx.errors:
        try:
            resolve_starts(model, lambda t: float(lib.effective(t)["bpm"]))
        except ValueError as e:
            cx.add("anchor_cycle", "plan.clips", str(e), "anchors must form a chain that starts at an at_sec clip")
    return (None, cx.errors) if cx.errors else (model, [])


def _intent(cx: _Ctx, v) -> dict:
    if v is None:
        return {}
    if not cx.obj(v, "plan.intent", INTENT_KEYS):
        return {}
    out = {}
    for k in ("beat_drift_ms_max", "low_end_overlap_max_sec"):
        if k in v and (n := cx.num(v[k], f"plan.intent.{k}", 0.0, None)) is not None:
            out[k] = n
    if "true_peak_max_dbtp" in v and (n := cx.num(v["true_peak_max_dbtp"], "plan.intent.true_peak_max_dbtp", -60.0, 6.0)) is not None:
        out["true_peak_max_dbtp"] = n
    for k in ("beat_drift_intended", "low_end_clash_intended"):
        if k in v:
            if isinstance(v[k], bool):
                out[k] = v[k]
            else:
                cx.add("wrong_type", f"plan.intent.{k}", "expected true or false")
    if "loudness_jump_lu" in v:
        j = v["loudness_jump_lu"]
        if isinstance(j, list) and len(j) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in j) and j[0] <= j[1]:
            out["loudness_jump_lu"] = [float(j[0]), float(j[1])]
        else:
            cx.add("wrong_type", "plan.intent.loudness_jump_lu", "expected [min, max] with min <= max", "e.g. [-1.5, 1.5]")
    if "silence_intended" in v:
        sv = v["silence_intended"]
        ok = isinstance(sv, list) and all(isinstance(r, list) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r) and r[0] < r[1] for r in sv)
        if ok:
            out["silence_intended"] = [[float(a), float(b)] for a, b in sv]
        else:
            cx.add("wrong_type", "plan.intent.silence_intended", "expected a list of [start_sec, end_sec]", "e.g. [[30.0, 31.0]]")
    return out


def validate_plan(plan, lib) -> list[dict]:
    return build_plan(plan, lib)[1]


def apply_plan(plan, lib) -> SetModel:
    model, errors = build_plan(plan, lib)
    if errors:
        raise PlanError(errors)
    return model


# ---------------------------------------------------------------- export
def _r(x: float) -> float:
    return float(x)  # no rounding: dense curve points must survive the round trip bit for bit


def export_plan(model: SetModel) -> dict:
    """The set as a plan; `apply_plan(export_plan(m))` rebuilds the same clips."""
    clips = []
    for c in model.clips:
        o: dict = {"id": c.id, "track": c.track, "label": c.label, "src_start": _r(c.src_start_beat / BAR), "bars": _r(c.length_beats / BAR)}
        st = c.start
        if "anchor" in st:
            o["start"] = {"over": st["anchor"], "at_bar": _r(st["anchor_beat"] / BAR)}
        else:
            o["start"] = {"at_sec": _r(st.get("at_sec", 0.0))}
        if st.get("offset_ms"):
            o["start"]["offset_ms"] = st["offset_ms"]
        for k, default in (("gain_db", 0.0), ("semitones", 0.0), ("key_lock", True), ("stems", False)):
            if getattr(c, k) != default:
                o[k] = getattr(c, k)
        env = lambda lane: {"points": [[_r(b / BAR), _r(v)] for b, v in lane.points]}
        if c.tempo.points:
            o["tempo"] = env(c.tempo)
        eq = {b: env(c.lanes[f"eq_{b}"]) for b in BANDS if c.lanes[f"eq_{b}"].points}
        if eq:
            o["eq"] = eq
        for k in ("filter", "volume"):
            if c.lanes[k].points:
                o[k] = env(c.lanes[k])
        sl = {s: env(c.lanes[ln]) for s, ln in STEM_LANES.items() if c.lanes[ln].points}
        if sl:
            o["stem_levels"] = sl
        if c.fx:
            o["fx"] = [{"id": f.id, "fx": f.fx, "at": _r(f.beat / BAR), "bars": _r(f.duration_beats / BAR), "beats": BEAT_LABELS[f.beats],
                        "wet": f.wet, **({"params": f.params} if f.params else {}), **({"stem": f.stem} if f.stem else {})} for f in c.fx]
        if c.loops:
            o["loops"] = [{"id": l.id, "at": _r(l.beat / BAR), "size_beats": BEAT_LABELS[l.size_beats], "bars": _r(l.length_beats / BAR),
                           "roll": l.roll} for l in c.loops]
        clips.append(o)
    out = {"version": PLAN_VERSION, "name": model.name, "notes": model.notes, "clips": clips}
    if model.intent:
        out["intent"] = model.intent
    return out
