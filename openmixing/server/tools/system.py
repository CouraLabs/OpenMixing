from __future__ import annotations

import importlib.util
import os
import shutil
import webbrowser

from ... import engine, mixing, validate
from ... import plan as planlib
from ...analysis import stems as stemlib
from ...analysis.analyze import AUDIO_EXT
from ...analysis.meta import AUDIO_EXTS
from ...core.units import BEAT_LABELS, SR
from ...dsp import fx as fxm
from ...dsp import sample, synth
from ...jobs import KINDS as JOB_KINDS
from ...jobs import MAX_CONCURRENT, STATES
from ...library import tags as taglib
from ...library.store import FIELDS as CRATE_FIELDS
from ...library.store import SORTS
from ...library.store import TAGS as HISTORY_TAGS
from ...mixing.techniques import TENSION_MAP
from ...model.lane import LANES
from ...model.rate import KINDS
from ...model.sampler import KINDS as SAMPLE_KINDS
from ...model.sampler import MAX_SECONDS, MAX_VOICES
from ...player import start
from ...setbuilder import PRESETS as ARC_PRESETS
from ...setbuilder import WEIGHTS as BUILD_WEIGHTS
from ...setbuilder.constraints import Constraints
from ..app import tool
from ..state import LOCK, STATE, cur, lib
from ..versions import versions
from ..views import jsonable


@tool()
def open_player(port: int = 8765, open_browser: bool = True) -> dict:
    """Start the local web player for the rendered set (play / seek, timeline, live deck panel with EQ / fx / loop state) and return its URL.
    Render the set first."""
    with LOCK:
        if STATE["ui"] is None:
            STATE["ui"] = start(lib(), port)
        url = STATE["ui"]["url"]
    if open_browser:
        webbrowser.open(url)
    rendered = any(lib().render_path(cur().name, f).exists() for f in ("mp3", "flac", "wav"))
    return {"url": url, "set": cur().name, "rendered": rendered, "hint": None if rendered else "call render first"}


@tool()
def project_info() -> dict:
    """Where OpenMixing is working: folder, state dir, current set, track count, dependencies status."""
    return {"folder": str(lib().root), "state_dir": str(lib().state), "current_set": cur().name, "tracks": len(lib().scan()),
            "optional": {k: bool(importlib.util.find_spec(k)) for k in ("beat_this", "essentia", "madmom", "torch", "demucs")},
            "online_keys": {k: bool(os.environ.get(k)) for k in ("LASTFM_API_KEY", "DISCOGS_TOKEN", "ACOUSTID_API_KEY")}}


@tool()
def capabilities() -> dict:
    """What this installation supports: file formats, stems, limits, effects, transition styles, curves, position formats, optional
    dependencies and all schema / analysis versions. Call it before planning if unsure whether a feature is available."""
    return jsonable({
        "versions": versions(),
        "input_formats": sorted(e.lstrip(".") for e in AUDIO_EXT),
        "formats": {e.lstrip("."): {"name": AUDIO_EXTS[e], "tags_writable": e in taglib.WRITABLE_EXT} for e in sorted(AUDIO_EXTS)},
        "tag_fields": {"file_and_sidecar": list(taglib.FILE_FIELDS), "sidecar_only": list(taglib.SIDECAR_ONLY), "colors": list(taglib.COLORS)},
        "output_formats": ["mp3", "flac", "wav"], "default_output_format": "mp3", "mp3_bitrate_kbps": engine.MP3_KBPS, "sample_rate": SR,
        "stems": {"available": stemlib.available(), "model": stemlib.MODEL, "names": list(stemlib.STEMS)},
        "max_lanes": None, "max_lanes_note": "unbounded; limited by memory",
        "fx": {n: {"mode": f.mode, "uses_beats": f.uses_beats, "params": f.params} for n, f in fxm.CATALOG.items()},
        "beat_divisions": list(BEAT_LABELS.values()), "max_fx_tail_sec": fxm.MAX_TAIL_S,
        "rate": {"kinds": list(KINDS), "pitch": "follow", "slip": True, "speed_range": planlib.RANGES["rate_speed"],
                 "beats_range": planlib.RANGES["rate_beats"]},
        "dynamics": {"chain_max_steps": planlib.MAX_CHAIN, "wet_points": True, "fx": ["compressor", "sidechain_pump"],
                     "duck": {"keys": ["self:<stem>", "<clip id>"], "cross_deck_extra_render": True}, "master_bus": "limiter only"},
        "generators": {"kinds": synth.describe(), "noise_colors": ["white", "pink", "brown"], "deterministic": "seeded from the trigger id",
                       "level": "relative to the deck's RMS over the preceding 4 beats + 9 dB (gain_db 0 = typical peak); -18 dBFS RMS if silent",
                       "roll_curves": ["log", "linear"]},
        "library": {"collections": True, "crates": {"fields": list(CRATE_FIELDS), "sorts": list(SORTS), "source": "sidecar data only"},
                    "history": {"tags": list(HISTORY_TAGS), "written_by": "record_play"},
                    "stored_in": [".openmixing/collections.json", ".openmixing/crates.json", ".openmixing/history.jsonl"]},
        "set_builder": {"tools": ["build_set", "explain_order"], "arc_presets": list(ARC_PRESETS), "weights": BUILD_WEIGHTS,
                        "constraints": list(Constraints.__dataclass_fields__), "efforts": ["normal", "high"],
                        "tempo_and_key_arcs": "off unless given; *_follows_energy derives them from the energy arc", "output": "plan, never applied"},
        "validation": {"intent_keys": list(validate.INTENT_KEYS), "tension_shapes": list(validate.SHAPES),
                       "vocal_collision": "measured from the vocals stem when the clip plays stems, else estimated from the vocal curve",
                       "harmonic_clash": "chroma roughness (method chroma_roughness) + Camelot key distance per overlap",
                       "tension_match": "1 s tension curve of the render vs a target; only computed when the intent declares tension_target"},
        "sampler": {"slot_kinds": list(SAMPLE_KINDS), "max_seconds": MAX_SECONDS, "max_voices": MAX_VOICES, "fits": list(sample.FITS),
                    "params": sample.PARAMS,
                    "read_from": [str(lib().root), str(lib().state / "samples")],
                    "stored_as": "content-addressed 48 kHz stereo wav in .openmixing/samples/"},
        "techniques": {"techniques": mixing.TECHNIQUES, "elements": list(mixing.ELEMENTS), "tension_map": TENSION_MAP, "group_tag": "tension:<id>",
                       "plan_field": "clips[].techniques", "gap_styles": list(mixing.GAP_STYLES)},
        "jobs": {"supported_kinds": list(JOB_KINDS), "max_concurrent": MAX_CONCURRENT, "states": list(STATES),
                 "cancel": "cooperative, per clip and per 60 s limiter block; temp files are removed",
                 "idempotency": "default key hashes set, params, versions and content; reused: true", "stored_in": ".openmixing/jobs.jsonl",
                 "tools": ["start_render", "job_status", "job_result", "list_jobs", "cancel_job"]},
        "reproducibility": {"record": "<render>.repro.json", "verify_levels": ["metrics", "bit_exact"], "default_verify": "metrics",
                            "tool": "verify_render", "fields": ["versions", "ffmpeg", "libs", "set", "plan", "params", "tracks", "stems", "samples",
                                                               "seeds", "output"]},
        "transition_styles": list(mixing.TEMPLATES), "curves": list(planlib.CURVES), "plan_ranges": planlib.RANGES,
        "mixer_lanes": {k: {"neutral": v[0], "min": v[1], "max": v[2], "unit": v[3]} for k, v in LANES.items()},
        "position_formats": ["16", "'16 bars'", "'64 beats'", "'95.5s'", "'cue:<name>'", "'section:<label>+N'"],
        "file_rules": {"read": str(lib().root), "write": str(lib().state)},
        "optional": {k: bool(importlib.util.find_spec(k)) for k in ("beat_this", "essentia", "madmom", "torch", "demucs", "pyacoustid")},
        "ffmpeg": bool(shutil.which("ffmpeg")),
    })
