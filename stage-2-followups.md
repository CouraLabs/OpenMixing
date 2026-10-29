# Stage 2 follow-ups

Issues and discoveries from the Stage 2 implementation (WP1-WP11). Full suite was green at 314 tests, `ruff check openmixing` clean. Items are grouped by priority.

## Verification not done

- [ ] Run the "ear" items of the stage-2.md acceptance checklist on real tracks in `tracks/` (vocal regions on 3 tracks, spinback click/pitch, `add_duck` naturalness, riser peak, `build_tension` result). Only analysis, key matching and `build_set` planning were smoke-tested on real tracks; no full real-track render.
- [ ] Time the `cancel_job` 2 s target on a 5-minute set (tests only cover temp-file cleanup).
- [ ] Re-run the full suite once more on the final tree (WP10 ran only `test_setbuilder.py` after its last edit; WP11 ran the full suite afterwards, so confirm nothing drifted).

## Quality gaps

- [ ] WP10 `build_set` trimming: tracks with a short entry-to-exit window can play only 8-24 bars on real tracks. Trimming is window-driven, not exit-window-driven as specced.
- [ ] WP1 vocal proxy is weakly selective (marks 55-60% of bars as vocal, confidence capped at 0.4). Decide whether to keep it or return `source: null` (stage-2.md open question 1).
- [ ] WP1 per-phrase keys are noisy on real tracks (e.g. Solao: 3A overall, 8A in most phrases, confidence 0.03-0.23) because chroma comes from the full mix. Use stems or harmonic-percussive chroma.
- [ ] WP8 vocal level for full-mix clips is estimated from the proxy curve and ignores loops; `get_region what=["vocal"]` still needs stems.
- [ ] WP7 `remove_technique` resets filter/bass/volume lanes across the technique window, wiping any manual automation there.
- [ ] WP7 reverb throw shortens the pre-drop gap (0.5 beats requested measures about 0.3).

## Spec deviations to confirm or close

- [ ] WP1: Schmitt-trigger vocal regions (0.25 in / 0.15 out) instead of 1-bar hysteresis; unnormalised high-band flux; tension components use a minimum span; `analyze_track` omits per-bar components and `chroma_bar` (use `get_region`); `candidates.VERSION` not bumped.
- [ ] WP2: `pitch` field of `RateEvent` not implemented (vinyl pitch-follows-speed only).
- [ ] WP3: `match_key` shifts the incoming clip by default; tuning fine-shift applied only at >= 10 cents; no dedicated key-fit ranking test in `candidates/rank.py`.
- [ ] WP4: duck keys are whole-deck for cross-deck; `self:<stem>` only on the target clip; tails of send effects inside a chain are not fed into later steps; no master bus compressor.
- [ ] WP5/WP6: generator and sample params are numeric (`color`, `tone`, `fit`, `choke`) instead of string kinds; riser/impact level is relative (RMS of previous 4 beats + 9 dB), no absolute level; 16-voice cap checked per clip, not per set; slots store absolute paths (set files not relocatable); no loop crossfade, no `root_key`/`start_sec`/`end_sec` on slots.
- [ ] WP7: technique lives in `mixing/techniques/`; `tension` is a technique parameter, not a `Clip` field; level mapping raised (filter 0.85, bass kill -60 dB, riser -3 dB); `gap_beats` not in `suggest_transitions` candidates.
- [ ] WP8: `tension_target` uses `start_sec`/`end_sec`, not `window_bars`; candidates do not emit a vocal-overlap `intent`; `render_variations` does not print predicted vs measured.
- [ ] WP9: `render` never records history (no `record_history` option); `lookup_online` genre is not a crate field.
- [ ] WP10: only `build_set` returns `infeasible`; `explain_order` returns violated constraint names.
- [ ] WP11: only `render` is a job kind (no jobs for `analyze_all` / `separate_stems`); cancel checked per clip and per limiter block; per-track `blake2b` in the repro record is the 6-byte content id.

## Housekeeping

- [ ] `ruff check tests` has 8 pre-existing errors in `tests/test_stems.py` and `tests/test_stage1_finish.py`.
- [ ] The stale `sed`/python patch for WP1 `analysis/tension.py` was not applied (WP1 already uses the scaled tension); delete it from any notes.
- [ ] Review and commit: all Stage 2 work sits uncommitted on top of the package refactor (large deletions/renames in `git status`).
- [ ] Stage-2 "Decisions for the user" 1-9 were answered with defaults (proxy vocals, vinyl-only pitch, master limiter only, sidecar-only crates, explicit `record_play`, metrics-level verify, samples in tracks folder and `.openmixing/samples/`). Confirm or change.
