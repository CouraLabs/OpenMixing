from __future__ import annotations

from ..core.units import BEAT_DIVISIONS, BEAT_LABELS
from ..dsp import sample, synth
from ..model.lane import STEM_LANES
from ..model.sampler import KINDS as SLOT_KINDS
from .ranges import MAX_CHAIN, num_range

BEATS = {"enum": [BEAT_LABELS[b] for b in BEAT_DIVISIONS], "default": "1"}
GROUP = {"type": "string", "description": "technique tag (tension:<id>) set by plan techniques; remove_technique deletes by it"}
LEN = {"type": ["number", "string"], "exclusiveMinimum": 0, "description": "bars or \"N beats\""}


def fx_schema(names: list[str]) -> dict:
    step = {"fx": {"enum": names}, "beats": BEATS, "wet": {**num_range("wet"), "default": 1.0},
            "params": {"type": "object", "description": "per-effect numbers, see list_fx"}}
    return {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["at", "bars"], "description": "give exactly one of fx / chain",
        "properties": {"id": {"type": "string"}, "group": GROUP, "at": {"$ref": "#/$defs/out_position"}, "bars": LEN, **step,
                       "chain": {"type": "array", "minItems": 1, "maxItems": MAX_CHAIN, "description":
                                 "effects in series over the same window; each step has its own wet, event wet scales all",
                                 "items": {"type": "object", "additionalProperties": False, "required": ["fx"], "properties": step}},
                       "wet_points": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}, "description":
                                      "[bar, wet] pairs from the event start, linear between; scales wet"},
                       "stem": {"enum": list(STEM_LANES), "description": "apply to this stem only (needs stems=true)"}}}}


def duck_schema() -> dict:
    return {"type": "array", "description": "sidechain ducks (plan 1.1): lower the target under a key signal inside a window", "items": {
        "type": "object", "additionalProperties": False, "required": ["target", "key", "at", "bars"], "properties": {
            "id": {"type": "string"}, "target": {"type": "string", "description": "clip id"},
            "key": {"type": "string", "description": "\"self:<stem>\" (a stem of the target, needs stem) or another clip id (its whole deck)"},
            "stem": {"enum": list(STEM_LANES), "description": "duck only this stem of the target (needs stems=true); default the whole deck"},
            "at": {"$ref": "#/$defs/out_position"}, "bars": LEN,
            **{k: num_range(k) for k in ("depth_db", "threshold_db", "attack_ms", "release_ms")}}}}


def trigger_schema() -> dict:
    return {"type": "array", "description": "synthesized transition material (plan 1.1), seeded from the id so renders repeat exactly; level is "
            "relative to the deck's recent RMS (gain_db 0 = the deck's typical peak)", "items": {
                "type": "object", "additionalProperties": False, "required": ["kind", "at"], "properties": {
                    "id": {"type": "string"}, "group": GROUP, "kind": {"enum": sample.KINDS},
                    "slot": {"type": "string", "description": "sampler slot, required for kind=sample"}, "at": {"$ref": "#/$defs/out_position"},
                    "bars": {**LEN, "default": 1.0}, "align": {"enum": ["start", "end"], "description":
                                                               "start: begins at `at` (hits); end: ends at `at` (risers)"},
                    "gain_db": {**num_range("gain_db"), "default": -6.0},
                    "params": {"type": "object", "description": "per-kind numbers, see list_generators",
                               "properties": {k: {"type": "number", "minimum": lo, "maximum": hi}
                                              for spec in [*(s["params"] for s in synth.KINDS.values()), sample.PARAMS]
                                              for k, (_, lo, hi) in spec.items()}}}}}


def sampler_schema() -> dict:
    return {"type": "object", "description": "sampler slots (plan 1.1): slot name -> a one-shot or loop file inside the tracks folder or "
            ".openmixing/samples/ (see load_sample); trigger it with clips[].triggers kind=sample", "additionalProperties": {
                "type": "object", "additionalProperties": False, "required": ["path"], "properties": {
                    "path": {"type": "string"}, "kind": {"enum": list(SLOT_KINDS), "default": "oneshot"}, "root_bpm": num_range("root_bpm"),
                    "gain_db": num_range("gain_db"), "pitch_semitones": num_range("sample_semitones"), "choke_group": {"type": "string"}}}}


def technique_schema(names: list[str], elements: list[str]) -> dict:
    return {"type": "array", "description": "named techniques (plan 1.1) expanded at plan time into lanes, fx, loops and triggers tagged "
            "group=tension:<id>; the exported plan holds those primitives, not this block", "items": {
                "type": "object", "additionalProperties": False, "required": ["technique", "resolve_at"], "properties": {
                    "technique": {"enum": names}, "id": {"type": "string"},
                    "resolve_at": {"$ref": "#/$defs/out_position", "description": "the resolution downbeat, e.g. \"section:drop\" or \"cue:drop\""},
                    "build_bars": {**LEN, "default": 8, "minimum": 1}, "elements": {"type": "array", "items": {"enum": elements}},
                    "peak": {**num_range("tension_level"), "default": 0.9}, "gap_beats": {**num_range("gap_beats"), "default": 0.5},
                    "impact": {"type": "boolean", "default": True}, "reverb_throw": {"type": "boolean", "default": True},
                    "respect_existing_gap": {"type": "boolean", "default": True,
                                             "description": "gap_beats becomes 0 when the track already has a gap before the section"},
                    "tension": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2}, "minItems": 2,
                                "description": "[[bar, level 0..1], ...] inside the clip; drives filter, bass kill, volume, riser, roll"}}}}
