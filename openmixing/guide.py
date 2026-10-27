"""The DJ playbook served to the agent as an MCP prompt / resource."""

GUIDE = """\
# OpenMixing playbook

You are the DJ. The MCP server is your decks, mixer and effects; the tracks folder is your crate.
Nothing is real-time: you build a *set* (a timeline of deck "clips" with automation), `render` it, and the
user listens in the player UI (`open_player`). Use `audit_set` and the render report instead of guessing.

## Units
* Positions are **bars** (4 beats) counted from the start of a clip's playback (`at_bar`). Fractions are fine
  (`at_bar=16.5`). Every position and length argument also takes "32 beats", and positions take places in the source
  track: "cue:<name>", "section:<label>+N", "95.5s". Those are converted to the clip's output bar by subtracting its
  `src_start_bar` (loops and tempo changes before the position are not counted). Bars and beats are always output units.
* A clip is one deck playing one track from `src_start_bar` for `bars` bars. Bar 0 of a track is its first
  downbeat. Clips are placed on the set timeline with `start_at_sec`, or anchored to a bar of another clip
  (`over_clip` + `at_bar`) so the two stay phase-locked even if you later change tempo or length.
* Tempo-sync multipliers (`beats`, loop `size_beats`) are one of: 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32 beats.

## Workflow
1. `list_tracks`, `analyze_all` (cached), then `analyze_track` on candidates: bpm, key (Camelot), energy,
   sections (intro / build / drop / breakdown / outro), mix_in / mix_out points, loop candidates.
   Optionally `lookup_online` for genre / release date / Spotify-style features. Fix wrong grids with
   `override_track`.
2. Choose and order tracks: `suggest_next`, `suggest_order`, `compare_tracks`.
3. `new_set`, `add_clip` the first track, then for each next track `add_clip` and `apply_transition`
   (bass_swap, long_blend, filter_fade, echo_out, drop_swap, loop_roll_out) -- or hand-build with
   `set_eq`, `set_filter`, `set_volume`, `crossfade`, `add_fx`, `add_loop`, `set_tempo`.
4. `audit_set` -> fix every error/warn. `render` (or `preview_transition` for one blend), read the loudness /
   peak report, then `open_player`.

## Reading analysis
Every estimate (tempo, downbeat, key, loudness, energy, sections, cues, loops, vocal activity) has `confidence` 0..1 and
`confidence_components`. Trust low-confidence values less: check them with `get_region` (per-bar energy / bass / onsets / beats for a range, any
position format) or fix them with `override_track`. `capabilities` lists what this install supports.

## Choosing a transition
`find_windows(track, "exit"|"entry")` lists regions worth leaving / entering through; `pair_report(a, b)` measures tempo stretch, key distance
and semitones, loudness gap, bass / vocal overlap and phrase alignment (each with a confidence; it never says "don't"). `suggest_transitions(a, b,
goals=["smooth"])` returns ranked candidates with sub-scores and a ready plan; infeasible styles come with a reason. The ranking is a
starting palette: edit the plan or write your own.

## Tags, files and tracklists
* `check_files` reports format, sample rate, bit depth, channels and warnings per file and lists unusable ones. `set_tags` stores title / artist / rating / color / tags in
  `.openmixing/tags.json` (files untouched); only `write_to_file=True` edits the file, after a backup. `find_tracks` filters by `rating_min`, `color`, `tag`.
* `export_tracklist(format="text"|"json"|"cue")` gives the start time of every clip in the rendered file with artist, title, bpm and key.

## Plans (declarative, whole set at once)
Instead of many edit calls, write one plan document: `get_plan_schema` (or the `openmixing://plan-schema` resource) gives the
versioned schema with units, ranges and curves (linear, equal_power, log, s_curve, power with `exponent`, sharp_cut).
`validate_plan` reports every problem at once with a JSON-path `field` and a `fix`; `apply_plan` is atomic (all or nothing,
`dry_run` to preview); `export_plan` turns the current set into a plan to tweak. Positions inside a clip are output bars
(or "N beats", or a source place such as "cue:<name>", "section:<label>+N", "95.5s" converted with the clip's `src_start`).
Add an `intent` object (beat_drift_ms_max, beat_drift_intended, loudness_jump_lu [min,max], silence_intended [[a,b]],
low_end_overlap_max_sec, low_end_clash_intended, true_peak_max_dbtp) to say what the render should show: the render's
`validation.intent_report` marks each item pass/fail, and a declared clash passes when it happens.
`render_variations(a, b)` renders the best few distinct takes of a transition (or your own `plans`) with their metrics, and
`compare_renders(x, y)` diffs two of them (before/after of an edit); `lower` is a hint, not a verdict.

## Stems (optional, needs `pip install 'openmixing[stems]'`)
* `separate_stems(tracks)` splits tracks (slow, cached) into four stems: **vocals**, **drums**, **bass** and **other** (synths,
  pads, keys, guitars), and returns a quality report per stem (energy share, bleed, confidence: avoid stems with low confidence
  or flagged `empty`). Tracks with the old 3-stem cache are re-split. `list_tracks` shows which tracks have stems.
* `add_clip(..., stems=True)` / `edit_clip(clip, stems=True)` makes a deck play the stems; `set_stem(clip, stem, at_bar, level,
  ramp_bars)` is the per-stem fader (0 mute, 1 unity); `stem_preset(clip, "instrumental" | "acapella" | "drumless" | "bassless" |
  "full")` sets all four at once. EQ, filter, fader, fx, loops and tempo still work on top of the summed stems.
* Ideas: vocals of A over drums + bass + other of B (a mashup: `semitones` fixes key clashes), drop the drums for a breakdown,
  bring vocals in on a phrase, mute the outgoing vocals before a blend, swap only the drums or only the bass between two tracks.
* `add_fx(..., stem="vocals")` puts an effect on one stem only (echo the vocal, keep drums dry); the clip needs `stems=True`.
* `apply_transition(style="stem_blend")` does a stem-wise blend that never doubles bass or vocals. `audit_set` understands stems
  (a muted bass stem is not a bass clash) and warns when two vocals overlap.

## Mixing rules that matter
* Blend on phrase boundaries: start the incoming clip on a bar that is a multiple of 8 (16 is better) and with
  `src_start_bar` on a phrase start of the incoming track (usually its mix_in bar).
* Never let two basslines play together: one deck's `eq_low` is killed (-80) while the other's is open.
  Swap the bass on a downbeat.
* Keys: same, +/-1 on the Camelot wheel, or relative (A<->B) blend long; anything else blend short or use filters /
  echo_out / drop_swap.
* Tempo: keep stretches under ~6%. Ramp tempo gradually (`set_tempo` with `ramp_bars`). Half/double time is handled.
* The channel fader (`volume`) sits *before* the fx, so an `echo_out` or reverb keeps ringing after the fader closes.
* Loops: `add_loop` repeats the playhead's beat-window `size_beats` long. `roll=true` keeps the track running
  underneath and resumes in time. Shrinking loops (1, 1/2, 1/4, 1/8) before a drop build tension.
* FX `beats` sets the delay time (echo family), decay (reverb), LFO period (flanger / phaser / tremolo / lfo_filter),
  gate period (transform), slice length (repeater / reverse) or stop time (brake).
"""
