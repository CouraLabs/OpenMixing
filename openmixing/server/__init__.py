from .app import main, mcp
from .state import LOCK, STATE, clip_of, cur, info, jobs, lib, save
from .tools.analysis import analyze_all, analyze_track, get_region, lookup_online, override_track
from .tools.clips import add_clip, edit_clip, remove_clip
from .tools.crates import (
                           collection_add,
                           collection_remove,
                           create_collection,
                           get_crate,
                           get_history,
                           list_collections,
                           record_play,
                           save_crate,
                           tag_track,
)
from .tools.ducks import add_duck, remove_duck
from .tools.fx import add_fx, add_loop, list_fx, remove_fx, remove_loop
from .tools.jobs import cancel_job, job_result, job_status, list_jobs, start_render, verify_render
from .tools.library import check_files, delete_cue, find_tracks, list_cues, list_tracks, set_cue, set_tags
from .tools.mixer import apply_transition, clear_automation, crossfade, set_eq, set_filter, set_volume
from .tools.pitch import match_key, set_pitch
from .tools.plans import apply_plan, export_plan, get_plan_schema, validate_plan
from .tools.rate import add_rate, remove_rate
from .tools.render import compare_renders, export_tracklist, preview_transition, render, render_variations
from .tools.samples import list_samples, load_sample, remove_sample, trigger_sample
from .tools.setbuild import build_set, explain_order
from .tools.sets import audit_set, get_set, list_sets, new_set, open_set
from .tools.stems import separate_stems, set_stem, stem_preset
from .tools.suggest import compare_tracks, find_windows, pair_report, suggest_next, suggest_order, suggest_transitions
from .tools.system import capabilities, open_player, project_info
from .tools.tempo import set_tempo, sync_clip
from .tools.tension import build_tension, list_techniques, remove_technique
from .tools.triggers import add_roll, add_trigger, list_generators, remove_trigger
