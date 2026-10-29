# OpenMixing

An MCP server (stdio) that lets an agent DJ a folder of tracks. Open your agent in the tracks folder, call the
tools, and build a **set**: a timeline of deck clips with EQ, DJ filter, volume, tempo, tempo-synced fx and loops.
`render` mixes it offline to MP3 (320 kbps CBR) by default, or FLAC/WAV with `format="flac"` / `"wav"`; the web UI (`open_player`) only plays the result.

## Install
```
python -m venv .venv && .venv/bin/pip install -e .[dev]
# optional: .[stems] (Demucs stem separation, pulls torch), .[beat] (beat_this beat tracking), .[fingerprint] (AcoustID); ffmpeg is needed for LUFS/true-peak and mp3
```

## MCP config (run from the tracks folder)
```json
{"mcpServers": {"openmixing": {"command": "/path/to/OpenMixing/.venv/bin/python", "args": ["-m", "openmixing"]}}}
```
State lives in `./.openmixing/` (analysis cache, sets, renders). The agent playbook is served as the
`openmixing://guide` resource and the `dj_set` prompt.

## Tools
- **Library / analysis**: `list_tracks`, `analyze_track` (bpm, beat grid, downbeat, key/Camelot, loudness, energy,
  sections, mix points, loop candidates, tags), `analyze_all`, `lookup_online` (MusicBrainz, Deezer, iTunes,
  ReccoBeats; Last.fm / Discogs / AcoustID with API keys), `override_track`, `set_cue` / `list_cues` / `delete_cue`, `find_tracks`, `compare_tracks`,
  `suggest_next`, `suggest_order` (fast greedy)
- **Set builder**: `build_set` (pool = all / crate / list; energy arc presets `warm_up_peak_cooldown` `build` `plateau` `wave` or points; tempo and key arcs optional and independent, or derived from the energy arc; constraints such as `must_include`, `start_with`, `max_key_distance`, `avoid_played_within_days`; beam search, local improvement, phrase-aligned trimming to `duration_min`; returns a validated plan and never applies it) and `explain_order` (cost breakdown of an existing set)
- **Stems**: `separate_stems` (Demucs htdemucs -> vocals / drums / bass / other with a per-stem quality report, cached in `.openmixing/stems/`), then
  `add_clip` / `edit_clip` with `stems=True`, `set_stem` per-stem faders and `stem_preset` (instrumental / acapella / drumless / bassless); `apply_transition` style `stem_blend`
- **Tags / files**: `set_tags` (title, artist, album, genre, year, comment, label, isrc, rating, color, tags, notes; sidecar by default, `write_to_file=True` opts in to writing the file after a backup), `check_files` (format report per file; MP3, FLAC, WAV, AIFF, AAC/M4A, OGG, Opus), `export_tracklist` (text / json / cue)
- **Library**: `create_collection` / `collection_add` / `collection_remove` / `list_collections`, `save_crate` / `get_crate` (smart crates: all/any rules over bpm, camelot, energy, duration, rating, color, tags, artist, title, genre, notes, play_count, last_played, has_stems; sidecar data only), `tag_track`, `record_play` / `get_history` (history is written only by `record_play`), `find_tracks(crate=...)`; stored in `.openmixing/`
- **Sets**: `list_sets`, `new_set`, `open_set`, `get_set`, `add_clip`, `edit_clip`, `remove_clip`
- **Mixer**: `set_eq` (low/mid/high, down to kill), `set_filter`, `set_volume`, `crossfade`, `set_tempo` (curved ramps), `set_pitch`, `match_key`,
  `sync_clip`, `clear_automation`, `apply_transition` (bass_swap, long_blend, filter_fade, echo_out, drop_swap,
  loop_roll_out)
- **FX / loops**: `list_fx`, `add_fx` (optionally `stem=` to hit one stem only), `remove_fx`, `add_loop`, `remove_loop`, at 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32 beats
- **Dynamics / chains**: `add_fx(chain=[{fx, beats, wet, params}, ...], wet_points=[[bar, wet], ...])`, `compressor` and `sidechain_pump` fx, `add_duck(target, key, at_bar, bars, depth_db, ...)` / `remove_duck` (key `self:<stem>` or a clip id); plan `fx[].chain`, `fx[].wet_points`, `ducks`
- **Generators**: `add_trigger(clip, kind, at_bar, bars, align, gain_db, params)` / `remove_trigger` / `list_generators` for synthesized `riser`, `downlifter`, `impact`, `crash`, `snare`, `noise` (post-fader, seeded from the trigger id so re-renders are identical, level relative to the deck); `add_roll` for an accelerating snare roll; plan `clips[].triggers`
- **Sampler**: `load_sample(slot, path, kind)` imports a one-shot or loop (tracks folder or `.openmixing/samples/` only, stored normalized and content-addressed), `trigger_sample(clip, slot, at_bar, bars, fit, choke ...)`, `list_samples`, `remove_sample`; loops follow the clip tempo; plan 1.1 top-level `sampler` and `clips[].triggers` with `kind="sample"`, `slot`
- **Tension**: `build_tension(clip, resolve_at, build_bars, elements, peak, gap_beats, tension)` ramps a high-pass, bass kill, volume swell, riser, snare roll and loop roll into a silent pre-drop gap, a reverb throw and an impact on the resolution downbeat (`section:drop`, `cue:` or a bar); it expands to ordinary events tagged `group=tension:<id>` (undo with `remove_technique`, plan field `clips[].techniques`, `list_techniques` shows the level mapping). `apply_transition(gap_beats=...)` adds the same breath to `echo_out` / `drop_swap`
- **Rate**: `add_rate` (spinback, brake, reverse, custom `[u, speed]` curve, `slip`), `remove_rate`; vinyl style, pitch follows speed; plan 1.1 `clips[].rate`
- **Transition planning**: `find_windows` (exit / entry regions), `pair_report` (tempo, key, loudness, bass / vocal overlap, phrase, each with confidence), `suggest_transitions` (ranked candidates with sub-scores, plan documents, infeasible reasons)
- **Plans**: `get_plan_schema`, `validate_plan`, `apply_plan` (atomic, `dry_run`), `export_plan`: describe a whole set as one versioned JSON document
- **Jobs and reproducibility**: `start_render` (background, idempotent: same set + params returns `reused: true`), `job_status`, `job_result`, `list_jobs`, `cancel_job` (cooperative, no temp files left); every render writes `<render>.repro.json` (versions, ffmpeg line, hashes, seeds, output hash) and `verify_render` re-renders from it (`metrics` default, `bit_exact`)
- **Output**: `audit_set`, `render`, `preview_transition`, `open_player`, `project_info`
- **Discovery / detail**: `capabilities` (formats, stems, fx, styles, curves, optional deps, versions), `get_region` (energy, bass, onsets, vocal, beats, sections, loops, cues, waveform for a range of bars)

Track ids are content hashes, so renaming or moving a file keeps its analysis. Positions accept bars, `"64 beats"`, `"95.5s"`,
`"cue:<name>"` and `"section:<label>+N"`. Tool errors come back as `{error: {code, field, message, fix}}`. Every `render` includes a
`validation` block (low-end overlap per deck, vocal collision, harmonic clash, optional tension match and pre-drop gap, intent report, clipping, unplanned silence, beat drift in ms, loudness jump across overlaps). Every analysis estimate has a `confidence` (0..1) and
`confidence_components`; responses carry a `versions` block. Files (tracks and samples) are read only inside the tracks folder or `.openmixing/samples/` and written only inside `.openmixing/` (the one exception is `set_tags(write_to_file=True)`, which edits the tags of the audio file you name, after a backup)
(`path_not_allowed` otherwise). `render_variations` renders distinct takes of a transition and `compare_renders` diffs two. Plans can declare an `intent`. See `CHECKLIST.md` for the roadmap.

Section boundaries are snapped to the 4-bar grid (`raw_start_bar` keeps the detected bar). When Demucs leaves the bass stem empty
(808 music), the low end is treated as living in the drums stem: `effective(...)["bass_carrier"]` reports `"drums_lowband"`.

See `stage-1-testing.md` for a manual test guide.

## License
GPL-3.0-or-later, see `LICENSE`.

## Limitations
Downbeat detection is a kick-energy heuristic; Last.fm/Discogs/AcoustID paths are untested without keys; no vocal
detection; Serato crate/marker import is not implemented.
