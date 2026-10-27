# OpenMixing

An MCP server (stdio) that lets an agent DJ a folder of tracks. Open your agent in the tracks folder, call the
tools, and build a **set**: a timeline of deck clips with EQ, DJ filter, volume, tempo, tempo-synced fx and loops.
`render` mixes it offline to FLAC/WAV/MP3; the web UI (`open_player`) only plays the result.

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
  `suggest_next`, `suggest_order`
- **Stems**: `separate_stems` (Demucs htdemucs -> vocals / drums / bass / other with a per-stem quality report, cached in `.openmixing/stems/`), then
  `add_clip` / `edit_clip` with `stems=True`, `set_stem` per-stem faders and `stem_preset` (instrumental / acapella / drumless / bassless); `apply_transition` style `stem_blend`
- **Tags / files**: `set_tags` (title, artist, album, genre, year, comment, label, isrc, rating, color, tags, notes; sidecar by default, `write_to_file=True` opts in to writing the file after a backup), `check_files` (format report per file; MP3, FLAC, WAV, AIFF, AAC/M4A, OGG, Opus), `export_tracklist` (text / json / cue)
- **Sets**: `list_sets`, `new_set`, `open_set`, `get_set`, `add_clip`, `edit_clip`, `remove_clip`
- **Mixer**: `set_eq` (low/mid/high, down to kill), `set_filter`, `set_volume`, `crossfade`, `set_tempo`,
  `sync_clip`, `clear_automation`, `apply_transition` (bass_swap, long_blend, filter_fade, echo_out, drop_swap,
  loop_roll_out)
- **FX / loops**: `list_fx`, `add_fx` (optionally `stem=` to hit one stem only), `remove_fx`, `add_loop`, `remove_loop`, at 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32 beats
- **Transition planning**: `find_windows` (exit / entry regions), `pair_report` (tempo, key, loudness, bass / vocal overlap, phrase, each with confidence), `suggest_transitions` (ranked candidates with sub-scores, plan documents, infeasible reasons)
- **Plans**: `get_plan_schema`, `validate_plan`, `apply_plan` (atomic, `dry_run`), `export_plan`: describe a whole set as one versioned JSON document
- **Output**: `audit_set`, `render`, `preview_transition`, `open_player`, `project_info`
- **Discovery / detail**: `capabilities` (formats, stems, fx, styles, curves, optional deps, versions), `get_region` (energy, bass, onsets, vocal, beats, sections, loops, cues, waveform for a range of bars)

Track ids are content hashes, so renaming or moving a file keeps its analysis. Positions accept bars, `"64 beats"`, `"95.5s"`,
`"cue:<name>"` and `"section:<label>+N"`. Tool errors come back as `{error: {code, field, message, fix}}`. Every `render` includes a
`validation` block (low-end overlap per deck, intent report, clipping, unplanned silence, beat drift in ms, loudness jump across overlaps). Every analysis estimate has a `confidence` (0..1) and
`confidence_components`; responses carry a `versions` block. Files are read only inside the tracks folder and written only inside `.openmixing/` (the one exception is `set_tags(write_to_file=True)`, which edits the tags of the audio file you name, after a backup)
(`path_not_allowed` otherwise). `render_variations` renders distinct takes of a transition and `compare_renders` diffs two. Plans can declare an `intent`. See `CHECKLIST.md` for the roadmap.

Section boundaries are snapped to the 4-bar grid (`raw_start_bar` keeps the detected bar). When Demucs leaves the bass stem empty
(808 music), the low end is treated as living in the drums stem: `effective(...)["bass_carrier"]` reports `"drums_lowband"`.

See `stage-1-testing.md` for a manual test guide.

## License
GPL-3.0-or-later, see `LICENSE`.

## Limitations
Downbeat detection is a kick-energy heuristic; Last.fm/Discogs/AcoustID paths are untested without keys; no vocal
detection; Serato crate/marker import is not implemented.
