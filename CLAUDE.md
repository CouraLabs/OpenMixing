# OpenMixing

MCP (stdio, FastMCP) server that lets an agent DJ a folder of tracks: analysis, beatmatching, EQ, loops, tempo-synced fx and stems,
rendered offline to a set file, plus a local web player for the render.

## Commands

```bash
.venv/bin/python -m pip install -e '.[dev]'          # install (extras: beat, stems, fingerprint)
.venv/bin/python -m pytest -q -p no:cacheprovider    # full suite, ~3-4 min
.venv/bin/python -m pytest -q tests/test_loops.py    # one file
.venv/bin/ruff check openmixing                      # lint (must stay clean, line length 150)
.venv/bin/ruff check openmixing --fix                # sort imports, drop unused ones
openmixing mcp --dir ~/Music/set                     # run the MCP server
openmixing ui --dir ~/Music/set                      # run only the web player
```

## Layout

Dependencies point downward only: `server` -> `plan` / `mixing` / `candidates` / `player` -> `engine` / `validate` / `library` /
`analysis` -> `model` / `dsp` -> `core`. Never import upward.

| Package | Responsibility |
|---|---|
| `core/` | units and position parsing (`units.py`, `BAR = 4.0`, `SR`), errors, audio decode, ffmpeg loudness, JSON file I/O |
| `model/` | set data: `Lane` automation, `FxEvent` / `LoopEvent`, `Clip`, `SetModel`, `TempoMap` (beat <-> seconds / samples) |
| `dsp/` | EQ, filter, fx, limiter, time-stretch |
| `analysis/` | per-track analysis (tempo, key, loudness, structure), tags, online lookup, region slices, Demucs stems |
| `library/` | `Library` = `Catalog` (files, ids, sets) + `TrackStore` (analysis cache, `effective`) + `StemStore` + `Annotations` (overrides, tags, cues) |
| `engine/` | rendering: `segments` (play / loop timeline of a clip), `deck` (one clip), `mixdown` (whole set), tracklist |
| `validate/` | render metrics and intent report |
| `mixing/` | compatibility, transition templates, set audit |
| `candidates/` | exit / entry windows, pair reports, ranked transition suggestions |
| `plan/` | declarative plan schema, validation (`Ctx` collects errors), build / apply / export |
| `player/` | `build_state` (JSON for the UI), waveform features, HTTP server `web.py`, `static/` (html, css, ES modules) |
| `server/` | FastMCP app, shared state (`LOCK`, `STATE`, `lib()`, `cur()`), `edit.automate`, views, one module per tool group in `tools/` |

`openmixing/server/__init__.py`, `openmixing/model/__init__.py`, etc. are facades: import from the package, not from deep modules, in tests.

## Code rules

- No code comments and no docstrings, except the docstring of an MCP `@tool()` / resource / prompt function (that text is the tool
  description the agent sees).
- YAGNI: no speculative options, aliases or wrappers. Delete dead code instead of keeping it "just in case".
- DRY: reuse the shared helpers before writing a new one:
  `core.jsonio.read_json/write_json`, `server.edit.automate` (lock + clip + lane writes + save), `Library.effective(ref)` (bpm,
  downbeat, gain, key, `n_bars`, stems with overrides applied), `server.state.at_of/len_of/conv_beat`,
  `analysis.online.http.provider` (error handling for online lookups), `dsp.fx.RENDERERS` (register a new effect there and in `CATALOG`).
- SOLID: one responsibility per module; keep files small (aim for under ~150 lines). Split by concern into a package with an
  `__init__.py` facade rather than growing a file.
- Imports at module top, relative inside the package, sorted by ruff (`I`). Lazy imports only to keep heavy optional dependencies
  (torch / demucs, MCP when running `ui`) off the import path.
- Match the surrounding style: short names, comprehensions, dict results from tools, `OpenMixingError(code, message, field, fix)` for
  user-facing errors.
- Front end: plain ES modules under `player/static/js/`, no build step, no framework. Shared mutable state lives in `store.js` (`st`);
  each module owns one panel (`timeline`, `zoom`, `decks`, `flags`, `transitions`). `web.py` only serves files inside `static/`.

## Domain rules

- Units: positions inside a clip are output beats from the clip start (`bar * BAR`). Tool arguments are bars or position strings
  (`"32 beats"`, `"cue:drop"`, `"section:breakdown+4"`, `"95.5s"`) converted with `Library.out_bars`.
- Track bar 0 is the first downbeat. The player shows positions 1-based as `bar.beat` of the source track (`28.1` .. `28.4`).
- Loops: `LoopEvent(beat, size_beats, length_beats, roll)` repeats the `size_beats` after `beat` for `length_beats`
  (`ceil(length / size)` passes). Without roll, playback continues right after the looped region (the track has advanced by one pass);
  with roll it resumes where it would have been. `engine.segments` is the single source of truth for this; the renderer, the
  waveform and the UI (`segs` in the player state, `position.js`) all follow it. `tests/test_loops.py` locks the behaviour.
- Never mutate the set outside `with LOCK:` and always `save()` after a change.

## Testing

- Every behaviour change gets a test in `tests/`. Use `tests/synth.py` and in-memory audio rather than real tracks.
- Run the full suite before finishing; it must be green. Baseline: 157 tests.
- For UI changes, render a small synthetic set and check it in a browser (`openmixing ui --dir <folder>`).
