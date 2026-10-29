from __future__ import annotations

from typing import Literal

from ... import plan as planlib
from ... import setbuilder
from ..app import tool
from ..state import cur, lib
from ..views import jsonable

Arc = list[list[float]] | Literal[*setbuilder.PRESETS] | None


@tool()
def build_set(pool: str | list[str] | None = None, duration_min: float = 60.0, energy_arc: Arc = "warm_up_peak_cooldown", tempo_arc: Arc = None,
              key_arc: list[list[float]] | None = None, tempo_follows_energy: bool = False, key_follows_energy: bool = False,
              constraints: dict | None = None, effort: Literal["normal", "high"] = "normal", seed: int = 0, weights: dict | None = None,
              goals: list[str] | dict[str, float] | None = None, include_plan: bool = True) -> dict:
    """Build an ordered, time-fitted set plan from a pool (default: all analyzed tracks; a crate name; or a list of tracks). Beam search over
    the order, then local improvement (effort="high" adds seeded annealing, never worse than normal), then each track is trimmed to whole
    phrases (entry window start, play_bars) so the set lasts `duration_min`. energy_arc: preset (warm_up_peak_cooldown, build, plateau, wave)
    or [[t_frac 0..1, level], ...] (levels 1-10; presets scale to the pool's energy range). tempo_arc ([[t_frac, bpm]] or preset scaled to the
    pool's bpm range) and key_arc ([[t_frac, camelot number 1-12]]) are off unless given; tempo_follows_energy / key_follows_energy derive
    them from the energy arc. constraints (hard unless noted): must_include, must_exclude, start_with, end_with, fixed_positions {index: track},
    must_not_follow [[a, b]], max_bpm_jump_percent, max_key_distance, min_gap_between_artist_repeats, avoid_played_within_days (history from
    record_play), max_tracks, only_with_stems, crate, duration_max (min), duration_tolerance_percent (warning, default 5), min_play_bars
    (default 32). weights: transition, energy, tempo, key. Returns the order with per-transition costs, arc trace vs target, the top
    alternatives, audit and a plan (validated) for apply_plan; it never changes the current set. When nothing satisfies the constraints it
    returns `infeasible` with the violated constraint names and the nearest order."""
    return jsonable(setbuilder.build(lib(), pool, duration_min, energy_arc, tempo_arc, key_arc, tempo_follows_energy, key_follows_energy, constraints,
                                     effort, seed, weights, goals, include_plan))


@tool()
def explain_order(set: str | None = None, plan: dict | None = None, energy_arc: Arc = "warm_up_peak_cooldown", tempo_arc: Arc = None,
                  key_arc: list[list[float]] | None = None, tempo_follows_energy: bool = False, key_follows_energy: bool = False,
                  constraints: dict | None = None, weights: dict | None = None) -> dict:
    """Cost breakdown of an existing set (default: the current one), a saved set by name, or a plan document, with the same cost function
    build_set optimises: per transition (compat, energy, tempo, key terms) against the arcs, total by term, and the violated constraints. Energy
    of each clip is taken from the part of the track it plays."""
    if plan is not None:
        model, errs = planlib.build_plan(plan, lib())
        if errs:
            return {"errors": errs}
    else:
        model = lib().load_set(set) if set else cur()
    return jsonable(setbuilder.explain(lib(), model, energy_arc, tempo_arc, key_arc, tempo_follows_energy, key_follows_energy, constraints, weights))
