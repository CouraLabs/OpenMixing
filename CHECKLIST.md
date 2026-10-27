# OpenMixing capability checklist

Source: the OpenMixing DJ Software Capability Spec (Oct 2026). `[x]` done, `[~]` partial, `[ ]` missing.
Priorities: P0 credible first release, P1 working DJ's toolkit, P2 creative depth. Update this file with every change.

## Stage 1 (P0)
- [x] Track registry: content-hash IDs (blake2b, cached by mtime/size), duplicate flag. Analysis cache is invalidated per `ANALYSIS_VERSION` and per file fingerprint (spec: invalidate on version change). A tag write pins the id (cache and stems stay valid); renaming a file after its bytes changed by a tag write gives a new id
- [x] Analysis: tempo, beatgrid, downbeat, phrases, key, LUFS, energy, bands. Every estimate (tempo, downbeat, key, loudness, energy, sections, cues, mix points, loops, vocal activity) carries `confidence` 0..1 plus `confidence_components` derived from its measurements. Cache is versioned (`ANALYSIS_VERSION` 3, see Real-music fixes). Per-bar onset density added
- [x] Timeline lanes with start in seconds or anchored to a bar of another clip
- [x] Flexible positions: bars, "N beats", `cue:name`, `section:label+N`, "95.5s" accepted by every clip tool (`at_bar`, `ramp_bars`, `bars`, `over_clip` anchors, eq / filter / volume / stem / fx / loop / tempo / crossfade / clear_automation) and by plans. Source units are converted with the clip's `src_start`. `apply_transition(at_bar, bars)` and plan `ramp_bars` / fx `bars` / loop `bars` also accept `"N beats"`
- [x] Tempo sync with pitch following (`key_lock=False`), `sync_clip`
- [x] Drift check: `validate.beat_drift` (ms, max/mean per overlapping pair), run on every render
- [x] Gain, volume, crossfade, 3-band EQ with kill, DJ filter, master limiter
- [x] Loops and loop extension done. Hot cues done (`set_cue`, `list_cues`, `delete_cue`, persisted per track). Cues store a color; the web player does not draw them (cosmetic)
- [x] Delay, reverb and filter-sweep fx (17 effects)
- [x] Stems: 4-stem Demucs separation (vocals / drums / bass / other, cache version 2, old 3-stem caches detected and re-split), per-stem level envelopes, stem fx (`add_fx(stem=...)` / plan `fx[].stem`: an effect on one stem only, e.g. echo the vocals and keep drums dry; effects without `stem` still run on the summed deck), instrumental / acapella via `stem_preset`, separation quality report (energy share, bleed, confidence per stem). Old sets and plans using `stem_drums_bass` / `stem_harmonic` still load
- [x] Compatible windows (`find_windows`), pair report with per-block confidence (`pair_report`), scored candidate transitions as plan documents with infeasible reasons (`suggest_transitions`, goals smooth/energetic/dramatic/experimental). Bass / vocal overlap are measured over the candidate's actual overlap and exposure scales with its length; a `blend` sub-score rewards a real overlap (so `smooth` prefers overlapped blends, and `loop_roll_out` / `drop_swap` win only under dramatic / experimental); windows can extend into adjacent bars of similar or lower energy (`extended_bars`). Vocal risk needs stems; style-level scores (harmony/bass exposure, blend, impact, novelty) are heuristic constants. Checked on 3 real pairs by numbers only, not by ear
- [x] Published plan schema (v1.0, `get_plan_schema` + `openmixing://plan-schema`), free-form plans (`apply_plan`, atomic, `dry_run`), plan validation (`validate_plan`: all errors at once with JSON-path field + fix), `export_plan` round trip. Plans cover clips, placement, tempo, gain/semitones, eq/filter/volume/stem automation with curves, fx, loops. Not yet in plans: master chain, sampler, cues as events
- [x] Transition render (`preview_transition`) and core validation: clipping, unplanned silence, loudness continuity, beat alignment
- [x] Compact responses, structured errors (`{code, field, message, fix}`), region queries (`get_region`, `analyze_track region=[a, b]`; per-bar arrays are downsampled to `max_points`)
- [x] Metadata: `set_tags` (title, artist, album, genre, year, comment, label, isrc, rating 0-5, color, free `tags`, notes). Default is sidecar-only (`.openmixing/tags.json`, your files are not touched); `write_to_file=True` also writes the container-capable fields (ID3 for MP3/WAV/AIFF, Vorbis comments for FLAC/OGG/Opus, MP4 atoms for M4A) after a backup copy in `.openmixing/backups/`; rating/color/tags/notes are always sidecar. `list_tracks` / `find_tracks` show and filter by rating, color, tag. Format report (`check_files`, `analyze_track` file_info.format_report, `capabilities.formats`): WAV, AIFF, FLAC, MP3, AAC/M4A, OGG, Opus; structured `unsupported_format` / `corrupt_file` errors. Tracklist and cue sheet: `export_tracklist(text|json|cue)`. Not done: raw `.aac` (ADTS) files cannot hold tags; ratings are not written into files
- [x] File handling rules: reads only inside the tracks folder (path refs outside and escaping symlinks rejected), writes only inside `.openmixing/` (render/set names sanitized, player audio route checked), error code `path_not_allowed`

## Stage 2 (P1)
- [~] Key lock, key shift (`semitones`) done. Key matching, tempo ramps (`set_tempo ramp_bars` exists, no curve choice)
- [~] Sections done. Vocal activity, tension features missing
- [~] Rolls, brake done. Slip behaviour, spinback, reverse playback, rate curves missing
- [~] Reverb, modulation, repeat, gate (transform), reverse, bitcrush done. Noise/risers, effect chains, compression missing
- [ ] Sampler slots and triggering
- [ ] Tension controls, resolution targets, pre-drop gap
- [~] Set building: `suggest_order`. No energy/tempo arcs, constraints, trimming
- [~] Full-mix render, preview. Missing: tracklist / cue sheet, jobs with cancel, reproducibility record
- [ ] Collections, smart crates, tags/notes, play history
- [x] Intent report (plan `intent`, pass/fail per declared item), before/after comparison (`compare_renders`), low-end overlap measured per deck on the render
- [ ] Vocal collision measured on the render, harmonic clash, tension curve match
- [~] Capability discovery (`capabilities`) and `versions` in analyze_track, get_set, render, find_windows, pair_report, suggest_transitions done. Long-running job API and idempotency missing

## Stage 3 (P2)
- [ ] Scratch patterns, cue juggling, beat juggling, slicer
- [ ] Stem swaps, acapella overlays, mashups, vocal chops
- [ ] Dissonance control, boldness, contrast discovery
- [ ] Key journeys, mood arcs, recurring motifs
- [ ] Serato/Rekordbox import and export

## Licensing
OpenMixing is GPL-3.0-or-later (non-commercial), so GPL dependencies (Pedalboard, Rubber Band) are fine. Still check separately: Demucs weights, beat-this weights, FFmpeg build.

- [x] Variations: `render_variations(a, b, count)` renders distinct-style takes (or your own plans) with metrics; `compare_renders` diffs two

## Next up
1. Candidate plans should carry an `intent` of their own (e.g. bass_swap expects little low-end overlap)

## Real-music fixes (tracks/ folder)
- [x] Exit windows skip the first phrase / intro, entry windows skip the outro (`find_windows`)
- [x] Section boundaries within 1 bar of the 4-bar grid are snapped onto it (`raw_start_bar` keeps the detected one); analysis version 3
- [ ] Downbeat detection is unchanged: the kick vote is ambiguous on four-on-the-floor tracks (downbeat confidence 0.01-0.12 on 8 of 11 real tracks). Needs ground truth to improve. Tracked in `downbeat-issue.md`
- [x] Empty bass stem: `stems.bass_carrier` / `low_stem`; `audit_set` bass clash and `stem_blend` fall back to the drums stem
