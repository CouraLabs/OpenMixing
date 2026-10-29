# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Hobbyist DJs and producers with a local folder of tracks who direct an AI agent to build a DJ set, then listen to and judge the result. They are at a desktop, usually with headphones or monitors, switching between the agent conversation and the player.

## Product Purpose
OpenMixing is an MCP (stdio) server that lets an agent DJ a folder of tracks: it analyzes them (tempo, beat grid, key, energy, sections), builds a set as a timeline of deck clips with EQ, filter, volume, tempo, tempo-synced fx and loops, and renders it offline to FLAC/WAV/MP3. The web player (`open_player`) only plays the rendered result. Its job is to let the user review and inspect the mix: hear it, and see exactly what the agent did (EQ kills, fx, loops, tempo, which decks are live) so they can judge it and give feedback.

## Positioning
The agent is the DJ and the user is the audience and critic. The player makes the agent's decisions legible, not just audible.

## Operating Context
- Runs locally against the user's own files; state lives in `./.openmixing/` (analysis cache, sets, renders).
- The player is served by `openmixing/ui/server.py`, polls for newly rendered sets, and plays the rendered file.
- Typical loop: agent builds or edits a set, renders, the user listens in the player, then gives feedback to the agent.

## Capabilities and Constraints
- Local-only: no cloud, accounts, or uploads.
- The UI is a single dependency-free `index.html` with no build step, served by `server.py`.
- Player shows: transport and seek, full-set timeline, zoomed waveforms of the current and next track with beat grid, per-deck readouts and jump pads, per-channel EQ/fader/filter state, and a track list.
- Terminology: set, clip, deck, bar, bass swap, kill, fx, loop, Camelot key.
- Undecided: whether the player will ever edit sets directly.

## Brand Commitments
Name is "OpenMixing". The existing wordmark and two-ring mark are in `openmixing/ui/index.html`.

## Evidence on Hand
Source and README in the repo. Test tracks live in `tracks/`. No user testimonials, usage data, or benchmarks exist; do not fabricate any.

## Product Principles
- Make the agent's decisions visible: anything that shapes the sound should be readable on screen.
- Judging by ear comes first; the interface supports listening and never competes with it.
- Stay local, light, and dependency-free.
- Work for a keyboard-and-mouse desktop session first, and stay usable on narrower windows.

## Accessibility & Inclusion
No specific standard established. Keyboard control of transport and seeking is expected, and meaning should not rely on color alone.

## License
GPL-3.0-or-later, open source.
