# Review 1 — deep pass, four angles

Date: 2025-10-28. Scope: full uncommitted restructuring (~10k lines). Four parallel review agents: test quality, concurrency/robustness, MCP tool ergonomics, performance + frontend. Every finding was verified by reading/running the code, not speculated. Supersedes nothing; round-1 findings (web.py sidecar leak, edit_clip negative bars, render snapshot race) are still open and re-confirmed by the concurrency pass.

This document has itself been audited by three verification agents (claim-by-claim re-check against the code). Net result: 0 claims wrong, 10 corrections applied below (marked **[corrected]**), 4 missed findings added (marked **[missed]**), and the fix order replaced.

## Suite status: RED

`pytest -q -p no:cacheprovider` → **1 failed, 316 passed in 13:03**.

- `tests/test_jobs.py:50` asserts `versions["analysis"] == 4`; `ANALYSIS_VERSION` is now 5. Same file's `test_capabilities_and_versions` (line 165) hardcodes `5`. Two hardcoded copies of one constant drifted apart. **[corrected: line was cited as 139]**
- Fix: `assert r["versions"]["analysis"] == ANALYSIS_VERSION` (import it, as `test_p0b.py:57` does).
- Docs drift: CLAUDE.md says "~3-4 min / 157 tests"; reality is 13 min / 316 tests (perf + 7-codec ffmpeg fixtures dominate). Update CLAUDE.md; consider a `-m "not perf"` fast lane.

## CRITICAL — data loss / corruption

**C1. `write_json` uses a fixed `.tmp` name — concurrent writers corrupt the final file** (`core/jsonio.py:15`)
Two writers to the same target open the *same* `<name>.tmp` and one renames garbage into place. Real scenarios: `analysis/<id>.json` written lock-free from the render thread while `analyze_track` analyzes the same track; `hashes.json` written by 4 `ProcessPoolExecutor` workers plus main in `scan()`.
Fix: unique tmp name (`.name.pid.uuid.tmp`) + `fsync` before `replace`.

**C2. `read_json` maps corrupt JSON → `default` silently** (`core/jsonio.py:8`)
A truncated `overrides.json` / `cues.json` / `tags.json` reads as `{}`: all manual BPM/key/gain overrides, cues and tags silently vanish from every render. Combined with C1 this is self-inflicted silent data loss. **[corrected]** `stems.json` reads as `None`, not `{}` (`stemstore.py:25` calls `read_json` with no default).
Fix: `FileNotFoundError` → default; malformed → raise or log loudly **for state files only** — analysis sidecars (`analysis/<id>.json`) are regenerable and must keep the soft default, or a truncated analysis file hard-fails every tool call with no recovery path.

**C3. Render sidecars written non-atomically while the player reads them** (`server/renders.py:45,52`, `tools/render.py:37`)
`publish` uses `write_text` while `/api/state` polls `read_bytes` — truncated JSON breaks the player until the next render; a crash mid-write corrupts `.validation.json` forever (`compare_renders` then dies on `json.loads`).
Fix: route all three through `write_json`.

**C4. MP3 render encodes directly to the final path** (`engine/mixdown.py:128`)
Crash mid-encode replaces the last good render with a partial file the player is actively reading. flac/wav already do tmp+replace.
Fix: encode to `.part`, then `replace`.

## HIGH — hangs, silent edits, races

Concurrency / server:

- **H1. `render` and `preview_transition` render the live model with no snapshot** (`tools/render.py:50`, `renders.py:20`). Async tools run concurrently; `cur()` in the worker thread can observe a half-mutated set, or a *different* set after `new_set`/`open_set`. Snapshot under LOCK like `start_render` does. (Round-1 finding #3, re-confirmed.)
- **H2. Concurrent renders collide on fixed tmp paths** (`mixdown.py:70,124`): `.raw.wav` / `.part.wav` are per-output constants; two renders to the same output interleave or unlink each other's files. Funnel through the JobManager or make tmp names unique.
- **H3. `decode()` loads whole files into memory, unbounded, untimed** (`core/decode.py:11`): a 2-hour decode ≈ 6 GB transient → OOM; no subprocess timeout anywhere on this path. Stream in blocks + `timeout=` + duration cap.
- **H4. `publish(meta)` runs on the event loop and decodes every track — plus the whole render** (`tools/render.py:53`). `build_state` → `track_features(path)` does a full `decode()` per clip **and** `band_envelope(meta["path"])` decodes the entire rendered mix (`waveform.py:26,33`): the whole MCP server freezes for tens of seconds to minutes. Wrap in `asyncio.to_thread` — the cheapest high-impact fix in this list.
- **H5. One corrupt line in `jobs.jsonl` permanently breaks all job tools** (`jobs/manager.py:47`): crash mid-append → `JobManager.__init__` raises until the file is hand-deleted. Skip/repair the trailing line; the file also grows forever.
- **H6. Non-daemon executor blocks exit; killed server orphans ffmpeg + tmp files** (`jobs/manager.py:45`): stdio close mid-render hangs shutdown until an hour-long render finishes. Daemon threads + startup sweep of `*.raw.wav`/`*.part*`.
- **H7. `hashes.json` is read-modify-written without LOCK across threads and processes** (`catalog.py:41` via `scan()`). **[corrected]** The original claim also named cues/overrides/tags — those are written only by *sync* tools, which FastMCP runs serialized on the event loop (`func_metadata.py:114`: sync `fn(...)` inline), so no concurrent writer exists; tags are additionally under LOCK. The real race is `scan()`'s `hashes.json`, written from the event loop, render/job worker threads, and 4 `ProcessPoolExecutor` workers simultaneously.
Fix: lock (or serialize) the `scan()` write path; the C1 unique-tmp fix makes the residual window harmless.

Tool layer (verified live against a synthetic library — the docstrings are unusually good and `guide.py` has zero drift; the traps are behavioral):

- **H8. `suggest_transitions` → `apply_plan` destroys the whole set** (`suggest.py:92`, `plans.py:27`). The suggestion says "pass a candidate's plan to apply_plan", but `apply_plan` *replaces* the set and the candidate plan is a 2-clip fragment. On a 10-clip set an agent following the documented path wipes 9 clips. Fix the note, or add a splice-by-clip-id merge path.
- **H9. Out-of-range/negative `at_bar` silently accepted by every lane tool** (`edit.py:automate`, `fx.py`, `ducks.py`, `rate.py`, `mixer.py:crossfade`). Live-verified: `set_eq(cid,"low",40.0,"kill")` returns success for bar 40 of a 31-bar clip — the kill never happens; negative `at_bar` poisons `value_at(0)` via interpolation (`core/units.py:70` rejects negative *source* positions but plain numbers bypass as output units). `add_trigger` validates start only (`triggers.py:58`); `add_roll` also checks end (`triggers.py:75`). **[corrected — fix shape]** Apply `at <= n_bars` only. Do *not* add `at + length <= n_bars`: `engine/segments.py:10,20` deliberately truncates events past clip end (`end = min(beat+length, length_beats)`) and `plan/events.py:46` / `plan/fx.py:94` deliberately accept overhanging fx/loops — a stricter tool rule would reject valid plans. Negative beats, however, must be rejected at `automate`.
- **H10. `remove_fx/loop/rate/trigger/duck` silently succeed on a wrong id** (live-verified: `remove_fx(cid,"fx_nope")` → success). A typo'd id looks applied. Raise `not_found` with valid ids, as `remove_clip` (`clips.py:95`) and `remove_sample` do — **[corrected]** `remove_sample` raises `unknown_slot`, not `not_found` (`samples.py:61`).

Tests (suite-wide):

- **H11. `save()` discipline is only partially verified.** **[corrected]** Sampler mutations are covered: `test_wp6.py:200,202` re-read the set file from disk after `remove_sample`/`load_sample`. Lane/fx/clip mutations are not: no test reloads after `set_eq`/`add_fx`/`add_clip`, so a tool that forgets `save()` there passes the suite.
- **H12. Player endpoints untested** except `/audio` traversal: `/api/state`, `/api/sets`, `/`, Range slicing (206) have zero coverage; `build_state`'s `segs`-follows-`engine.segments` invariant is unlocked too.
- **H13. Loop edge semantics unlocked**: `test_loops.py` covers only exact multiples. Missing: length not a multiple of size (partial last pass), loop truncated at clip end.
- **H14. Timeline redraws the whole canvas every animation frame, even paused** (`timeline.js:88,56,65`, `main.js:71`). Every `clipBox` runs one lane-interpolation `val()` per pixel (`timeline.js:47`) plus `fader` per 2 px; a 60-min set ≈ **16k** calls/frame (10.8k lowKills + 5.4k fader; the original 22k figure was ~35% high) + ~11M px clear at dpr 2 → 30–200 ms/frame. Fix: static offscreen layer (grid + boxes + waveform), blit + playhead per frame, gate on pause/dirty.
- **H15. `Catalog.scan()` runs a full directory walk per `resolve()`, and `info()` is called ~5–6× per clip during render** (`catalog.py:21,84`, `tracks.py:74`, `state.py:52`). Verified count in `mixdown.py`: `resolve_starts`, 2× TempoMap in `timeline()`/work loop, `load_audio`, `meters.add`, +1 per duck event → 5–6 `info()` per clip, each = 2 directory walks + analysis-JSON read. ~0.5 s per `info()` on a 1,000-track library → minutes added per render, paid by every tool call too. **[corrected — fix shape]** Memoize, but the naive `(track_id, fingerprint)` key is unsafe: `override_track` changes bpm/key/gain in `overrides.json` with no fingerprint change, `analyze(force=True)`/`separate()` rewrite sidecars under the same audio fingerprint, and memoized `scan()` hides files added mid-session. The key must include the override version and analysis-file mtime, or invalidate `scan()` on directory mtime. **[missed]** Related same-class cost the memoize fix wouldn't cover: `crates.py:11` `brief()` is an O(n) scan per track → `list_collections(name=…)` is O(n²).
- **H16. 3-second poll fetches and JSON-parses the full multi-MB state every tick** (`main.js:16,23,89`). Verified: `load()` fetches and parses `/api/state` *before* the `key === st.version` early-return, so download+parse happens every 3 s forever, even idle; `document.hidden` handling is absent (grep over all player JS). Add a cheap `/api/version` endpoint; skip polling when hidden.

## MEDIUM

Concurrency/robustness:

- `cur()` fallback can overwrite an existing set file from a truncated `current_set` (`state.py:36`); make `current_set` writes atomic and never overwrite from the fallback.
- LOCK held across heavy I/O: `lib().analyze()` inside LOCK (`tools/tension.py:28`), analysis inside `add_clip`/`edit_clip` (`clips.py:48,78`). Resolve before locking. (tension re-confirmed from round 1.)
- `separate()` unlinks `stems.json` then writes flacs one by one: a render in that window fails or silently plays the full mix; two concurrent separates corrupt stems (`stemstore.py:39`). Tmp dir + rename, per-track lock.
- No timeouts on ffmpeg/ffprobe anywhere except fpcalc; `cancel` has no teeth after the last checkpoint (`decode.py`, `meta.py`, `measure.py`, `mixdown.py`).
- `web.py` audio endpoint buffers whole ranges in memory (`bytes=0-` on a 1-hour mp3 ≈ 150 MB/request); `bytes=5-2` reads to EOF; no try/except on handlers (`web.py:71`).
- Lazy-init races on `lib()`/`cur()`/`jobs()` (`state.py:17-38`): two JobManagers → duplicate renders. Init under LOCK or at startup.

Tools:

- Set-name param spelled three ways: `set` (record_play, explain_order) vs `set_name` (export_tracklist) → rename all to `set_name`.
- Pair/partner params spelled four ways (`from_track/to_track`, `a/b`, `out_clip/in_clip`, `to_clip`) and `limit` vs `count`; standardize.
- Half the tools raise bare `ValueError` → code `invalid_value` with the generic "check the parameter ranges" fix and null `field` (stems.py:28,52, fx.py:34,70, clips.py:24, mixer.py:22). The taxonomy is coherent — apply it.
- `set_filter`/`set_volume` silently clamp while `set_eq` raises for the same class of mistake (`mixer.py:21,31,37`). Pick raise-with-`out_of_range`.
- `add_fx(chain=…)` silently discards `fx`/`beats`/`params` (`fx.py:25`); echo the mode used or reject the combo.
- `suggest_next`/`suggest_order` silently skip unanalyzed tracks — agents read the short list as "nothing mixes well"; add one docstring sentence.
- `Pos` schema description is wrong for `add_clip`/`edit_clip.at_bar` (bar of `over_clip`, not the placed clip) — separate annotation.
- No undo for destructive edits (`apply_plan` replace, `clear_automation(lane="all")`); nothing tells the agent to `export_plan` first.
- **[missed]** `set_cue` accepts any bar with no range check (`library.py:33` → `annotations.py:89`): a cue at bar 1000 of a 32-bar track silently persists — same silent-out-of-range class as H9, and cues feed `cue:<name>` positions into every lane tool.
- **[missed]** `clear_automation` range not validated (`mixer.py:59-77`): `from_bar=100` on a 31-bar clip "succeeds" clearing nothing; `from_bar > to_bar` reaches `Lane.clear(a, b)` (`lane.py:61`) which removes no points but appends two bogus neutral points.
- **[missed]** `delete_cue` raises bare `KeyError` (`annotations.py:98`) → `not_found` with generic fix and null `field` — same taxonomy gap as the ValueError item above.

Performance:

- `np.convolve` moving average in loudness is O(n×480): 1.2 s per 6-min track, 13 s per hour (`spectral.py:39`). **[corrected]** `uniform_filter1d` is not drop-in: `valid[i] == uniform_filter1d(...)[i+240]` — the replacement must slice `u[240::480]` and trim to `n-479`, or every silence boundary shifts one 10 ms frame. Measured speedup: 1.30 s → 0.13 s (10×, not the originally claimed 25×).
- `build_state` re-decodes every track with ffmpeg at every render end instead of persisting `track_features` (`state.py:14`, `waveform.py:47`) — ~15–30 s appended to every render for 12 tracks.
- Deck-overview canvases cached without bound on clip objects (`decks.js:41`) — ~100–150 MB on large sets.
- Timeline canvas height unbounded → blank canvas over browser limits on 200-clip sets (`timeline.js:20`); virtualize rows.
- `$('time').innerHTML` reparse 60×/s (`main.js:41`); O(n) per-frame class toggles over all clip rows (`main.js:44`).

Tests:

- `tests/test_plan.py:87` sets `STATE["lib"]/["set"]` with no teardown — passes only because `test_perf.py` runs next alphabetically. **[corrected]** no conftest.py exists at all.
- Queued jobs never observed to *run* (serialization untested); `test_verify_without_record` asserts type but not `.code`.
- `test_core.py:38` asserts catalog membership, not behavior, under a rejection-test name.
- `test_perf.py`'s RSS assertion uses cumulative `ru_maxrss` (depends on prior tests); a 1-file library structurally can't catch the scan() cost.

## LOW

- Dead code: `_grid_score` (`analysis/analyze.py:12`), `JobManager.report` (`jobs/manager.py:110`), dead lookups in `validate/intent.py:29` and `candidates/pair.py:23`, redundant clip lookup `mixing/audit.py:84`, both-arm-`"COMM"` conditional `library/tags.py:71`.
- `analysis/meta.py:10` imports upward from `library.tags` — move the extension constants into `core` (round-1 finding).
- `core/errors.py:7` imports MCP eagerly, defeating the lazy-import rule on the `ui` path.
- `dsp/stretch.py:44` writes through a caller's ndarray (`np.asarray` view) — latent shared-array bug; copy first.
- `core/measure.py:11` — ffmpeg failure → `rfind` = −1 → metrics silently `None`; also `decode.py:17` mislabels empty ffprobe output as `invalid_value`.
- `library/tags.py:74` — unguarded `f.tags` in the wav/aiff branch (None check exists only in the VORBIS branch); mutagen tag writes are in-place with no forced backup.
- `library/catalog.py` never prunes deleted files from `hashes.json`; `scan()` aborts the whole walk on one `stat()` race.
- `add_clip(bars=100)` on a 31-bar track accepted; `bars="cue:drop"` accepted as a *duration* (`clips.py:43,52`) — route lengths through `len_of`.
- `add_fx`/`add_loop` accept `at_bar` past clip end (plan validation rejects the same input — inconsistent).
- `export_tracklist` invents `audio_file` for an unrendered set; `remove_trigger` returns a count where siblings return id lists; `capabilities` duplicates four `list_*` payloads that will drift.
- `override_track` shift example reads directionally backwards (positive = later, example implies the opposite).
- `render_variations(a, b)` uses `a/b` while `crossfade` uses `out_clip/in_clip` for the same clips.
- `server/tools/analysis.py` (174 lines) over the ~150-line aim — split regions out.
- Frontend: dead CSS rules (`.pad.empty`, `.eq .tr`), ~20 `querySelector` calls per frame in `updateDeck`/`updateStrip` (cache refs at mount), full-size zero-vocals buffer allocation in `engine/deck.py:81`.
- Deep-module imports in tests where facades exist (`test_setbuilder.py:8`, `test_wp5.py:14`, `test_rate.py:16`, `test_library_store.py:11`); `versions`/`expand_window`/`advance` are not re-exported by any facade.
- LOCK is never exercised from two threads in tests.

## Verified clean (do not re-litigate)

- Loop semantics in `engine/segments` exactly match `test_loops.py`; all consumers (deck, waveform, state, position.js) follow it.
- `mixdown` streams through `raw.wav` — the whole set is never in RAM; **[corrected]** peak is materially higher than "one clip + 12 s tail": `AudioCache(size=6)` (`engine/cache.py`) holds up to 6 fully-decoded source tracks (~760 MB for 6-min tracks) and `acc` accumulates all simultaneously-overlapping clips between flushes — bounded, but ~6× the stated figure; limiter, stretch, EQ block loops are all vectorized or block-wise; no O(n²) in analysis.
- Audio served by disk range requests, not re-render; no JS memory leaks found (listeners attach once); zoom math correct.
- `SetModel.save`, flac/wav renders, tmp+rename patterns are same-directory atomic; JobManager marks orphans `lost` on restart; no lock-order inversion; path traversal properly blocked in `web.py`.
- 88 registered tools, no true duplicates, coherent error taxonomy (`not_found, invalid_value, invalid_position, invalid_range, out_of_range, wrong_type, bad_key, stems_off, unknown_slot, slot_in_use, unknown_param, path_not_allowed`) — unevenly applied, not absent.
- Docstring quality and guide.py↔registry consistency: zero drift.

## Suggested fix order **[replaced — the original order omitted 5 of 15 HIGH findings and inverted two test-before-change dependencies]**

1. Make the suite green: `test_jobs.py:50` → compare against `ANALYSIS_VERSION` (and dedupe with `test_jobs.py:165`).
2. `core/jsonio.py` C1+C2 with the regenerable-vs-state distinction (analysis sidecars stay soft, `overrides`/`cues`/`tags`/`stems` become strict) — **plus H5** (jobs.jsonl trailing-line repair), which lives in the same corrupt-input neighborhood but is read by raw `json.loads`, not `read_json`.
3. Cheap isolated fixes: **H4** (`asyncio.to_thread(publish, meta)` at `tools/render.py` — one line), **C4** (mp3 encode to `.part` + replace), **H2** (unique tmp suffixes or funnel through JobManager).
4. Test net before changing behavior: H11 (save-discipline reload test for lane/fx/clip tools) + H12 (player endpoint tests, including Range) — the endpoint contract must be locked before step 7 changes `/api/state` usage.
5. Tool traps: H8 (apply_plan note or merge path), H10 (remove_* not_found), H9 with plan-consistent semantics (`at <= n_bars` only, negative rejected in `automate`), plus the missed items (set_cue range check, clear_automation range, delete_cue error shape).
6. Server races: H1 (snapshot in render/preview_transition), C3 (sidecars via write_json), H7 (scan/hashes lock), H6 (daemon executor + tmp sweep).
7. Perf, largest last: H14 (offscreen timeline layer), H3 (streaming decode + timeouts), H15 (memoize with override/analysis-aware invalidation), H16 (`/api/version`, after H12 tests exist), M2 (persist track_features).

## Audit trail

Three verification agents re-checked this document claim-by-claim against the code. Verdicts: 10/11 concurrency claims CONFIRMED (H7 narrowed to `hashes.json` — the cues/overrides/tags leg was unreachable because sync tools run serialized on the event loop); all tool-layer claims CONFIRMED (H10's cited sibling raises `unknown_slot`, not `not_found`; H11 overstated — sampler save *is* tested); all perf/frontend claims CONFIRMED (H14's call count ~35% high; H15's memoization key unsafe as specified; the uniform_filter1d fix needs the `u[240::480]` alignment and yields 10×, not 25×); 0 claims WRONG. Four findings were added as **[missed]** (set_cue range, clear_automation range + bogus neutral points, delete_cue bare KeyError, crates `brief()` O(n²)).
