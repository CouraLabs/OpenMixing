from __future__ import annotations

import re

from ..core.units import BEAT_DIVISIONS, BEAT_LABELS
from ..dsp import fx as fxm
from ..mixing.techniques import CATALOG, ELEMENTS
from ..model.curves import CURVES
from ..model.lane import STEM_LANES
from ..model.rate import KINDS
from .fxschema import GROUP, duck_schema, fx_schema, sampler_schema, technique_schema, trigger_schema
from .ranges import num_range

PLAN_BASE_VERSION = "1.0"
PLAN_VERSION = "1.1"
PLAN_VERSIONS = (PLAN_BASE_VERSION, PLAN_VERSION)
BANDS = ("low", "mid", "high")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class PlanError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__(f"plan has {len(errors)} error(s): " + "; ".join(f"{e['field']}: {e['message']}" for e in errors[:5]))
        self.errors = errors


def _event_schema(value_key: str, value: dict, desc: str) -> dict:
    return {"description": desc, "oneOf": [
        {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["at", value_key], "properties": {
            "at": {"$ref": "#/$defs/out_position"}, value_key: value,
            "ramp_bars": {"type": ["number", "string"], "minimum": 0, "maximum": 256, "default": 0,
                          "description": "bars or \"N beats\"; 0 snaps (de-clicked ~10 ms)"},
            "curve": {"enum": list(CURVES), "default": "linear"},
            "exponent": {"type": "number", "minimum": 0.1, "maximum": 10, "description": "k of the u^k shape, curve=power only"}}}},
        {"type": "object", "additionalProperties": False, "required": ["points"], "properties": {
            "points": {"type": "array", "items": {"type": "array", "prefixItems": [{"$ref": "#/$defs/out_position"}, value],
                                                  "minItems": 2, "maxItems": 2}}},
         "description": "raw envelope: [at, value] pairs, linear between points"}]}


def plan_schema() -> dict:
    eq_val = {"oneOf": [num_range("eq"), {"const": "kill"}], "description": "dB, -80 = kill, 0 = flat"}
    fx_names = list(fxm.CATALOG)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"openmixing://plan-schema/{PLAN_VERSION}",
        "title": "OpenMixing set plan", "version": PLAN_VERSION, "type": "object", "additionalProperties": False,
        "required": ["version", "clips"],
        "properties": {
            "version": {"enum": list(PLAN_VERSIONS), "description": "1.1 adds rate, triggers, fx chains and ducks; 1.0 plans still load"},
            "name": {"type": "string"}, "notes": {"type": "string"}, "intent": {"$ref": "#/$defs/intent"},
            "clips": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/clip"}},
            "ducks": duck_schema(), "sampler": sampler_schema()},
        "$defs": {
            "intent": {"type": "object", "additionalProperties": False, "description":
                       "What you expect the render to show. The render's validation.intent_report marks each declared item pass or fail; "
                       "a declared clash passes when it happened.", "properties": {
                           "beat_drift_ms_max": {"type": "number", "minimum": 0}, "beat_drift_intended": {"type": "boolean"},
                           "loudness_jump_lu": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                                                "description": "[min, max] LU change across every overlap"},
                           "silence_intended": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                                                           "maxItems": 2},
                                                "description": "[start_sec, end_sec] on the set timeline"},
                           "low_end_overlap_max_sec": {"type": "number", "minimum": 0}, "low_end_clash_intended": {"type": "boolean"},
                           "true_peak_max_dbtp": {"type": "number"},
                           "vocal_overlap_max_sec": {"type": "number", "minimum": 0}, "vocal_clash_intended": {"type": "boolean"},
                           "harmonic_dissonance_max": {"type": "number", "minimum": 0, "maximum": 1}, "harmonic_clash_intended": {"type": "boolean"},
                           "pre_drop_gap_min_beats": {"type": "number", "minimum": 0},
                           "tension_target": {"type": "object", "description": "Rendered tension curve (1 s windows) compared with a target "
                                              "by Pearson correlation. Exactly one of shape / points; start_sec / end_sec on the set timeline.",
                                              "properties": {"shape": {"enum": ["rising", "falling", "build_release", "plateau"]},
                                                             "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                                                                                   "minItems": 2, "maxItems": 2}},
                                                             "min_correlation": {"type": "number", "minimum": -1, "maximum": 1},
                                                             "start_sec": {"type": "number", "minimum": 0},
                                                             "end_sec": {"type": "number", "minimum": 0}}}}},
            "out_position": {"description": "Position inside this clip's playback: output bars (bar 0 = clip start), \"N beats\", "
                                            "or a place in the source track (\"cue:<name>\", \"section:<label>+N\", \"95.5s\"), "
                                            "converted with the clip's src_start.",
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
                            "at_sec": {"type": "number", "minimum": 0}, "offset_ms": num_range("offset_ms")}},
                        {"type": "object", "additionalProperties": False, "required": ["over", "at_bar"], "properties": {
                            "over": {"type": "string", "description": "id of another clip in this plan"},
                            "at_bar": {"$ref": "#/$defs/out_position"}, "offset_ms": num_range("offset_ms")}}]},
                    "bpm": {**num_range("bpm"), "description": "constant playback tempo (default: native)"},
                    "tempo": _event_schema("bpm", num_range("bpm"), "tempo moves (pitch-fader), needs no bpm"),
                    "gain_db": num_range("gain_db"), "semitones": num_range("semitones"),
                    "pitch": _event_schema("semitones", num_range("semitones"), "pitch lane added to semitones; glides in <= 0.25 st steps"),
                    "key_match": {"type": "object", "additionalProperties": False, "required": ["to"], "properties": {
                        "to": {"type": "string", "description": "id of the clip whose key to match; resolved to semitones when the plan is built"},
                        "mode": {"enum": ["nearest_compatible", "exact"], "default": "nearest_compatible"},
                        "max_shift": {"type": "integer", "minimum": 0, "maximum": 12, "default": 3}}},
                    "key_lock": {"type": "boolean", "default": True}, "stems": {"type": "boolean", "default": False},
                    "eq": {"type": "object", "additionalProperties": False,
                           "properties": {b: _event_schema("gain_db", eq_val, f"{b} band") for b in BANDS}},
                    "filter": _event_schema("value", num_range("filter"), "-1 low-pass closed, 0 off, +1 high-pass closed"),
                    "volume": _event_schema("level", num_range("volume"), "channel fader, 1 = unity"),
                    "stem_levels": {"type": "object", "additionalProperties": False, "description": "needs stems=true and separate_stems",
                                    "properties": {s: _event_schema("level", num_range("stem"), f"{s} stem fader") for s in STEM_LANES}},
                    "fx": fx_schema(fx_names),
                    "triggers": trigger_schema(), "techniques": technique_schema(list(CATALOG), list(ELEMENTS)),
                    "loops": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["at", "size_beats"],
                              "properties": {"id": {"type": "string"}, "group": GROUP, "at": {"$ref": "#/$defs/out_position"},
                                             "size_beats": {"enum": [BEAT_LABELS[b] for b in BEAT_DIVISIONS]},
                                             "bars": {"type": ["number", "string"], "exclusiveMinimum": 0},
                                             "repeats": {"type": "integer", "minimum": 1},
                                             "roll": {"type": "boolean", "default": False}},
                              "description": "give exactly one of bars / repeats; loops may not overlap"},
                    "rate": {"type": "array", "description": "playback-rate events (plan 1.1): vinyl-style, pitch follows speed; may not overlap "
                             "each other or a loop. slip=true is the loop roll behaviour: the track keeps running underneath and rejoins in time",
                             "items": {"type": "object", "additionalProperties": False, "required": ["at", "bars"], "properties": {
                                 "id": {"type": "string"}, "kind": {"enum": list(KINDS), "default": "custom",
                                                                    "description": "preset curve when points are omitted; a label otherwise"},
                                 "at": {"$ref": "#/$defs/out_position"},
                                 "bars": {"type": ["number", "string"], "exclusiveMinimum": 0,
                                          "description": "bars or \"N beats\" (1/16 .. 32 beats)"},
                                 "points": {"type": "array", "minItems": 2, "items": {"type": "array", "minItems": 2, "maxItems": 2,
                                            "prefixItems": [{"type": "number", "minimum": 0, "maximum": 1}, num_range("rate_speed")]},
                                            "description": "[u, speed]: u 0..1 over the event, speed 1 normal, 0 stopped, negative backwards"},
                                 "slip": {"type": "boolean", "default": False}}}}}}}},
        "curves": {"linear": "straight ramp", "equal_power": "sin/cos fade (constant power when one rises while the other falls)",
                   "log": "log10(1+9u): fast start, slow end (mirrored when falling)", "s_curve": "cosine ease in/out",
                   "power": "u^exponent: slow start, surge at the end when exponent > 1 (tension builds)",
                   "sharp_cut": "hold, then jump at the end of the ramp"},
    }
