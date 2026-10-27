# Downbeat detection is unreliable

Status: open. Found 2026-10-27 while testing on the real tracks in `tracks/`.

## Problem
`first_downbeat` comes from a kick-energy heuristic (`openmixing/analysis/analyze.py`, `downbeat()`). Confidence is below 0.35 on 8 of 11 test tracks. A downbeat that is one beat off shifts every bar-based cue, loop, phrase window and transition by a beat.

| Track | First downbeat (s) | Confidence |
|---|---|---|
| Ana Julia | 0.504 | 0.33 |
| O Sol | 0.515 | 0.86 |
| Smack Yo' | 0.009 | 0.12 |
| Solao | 0.031 | 0.08 |
| The Weekend | 1.597 | 0.01 |
| Trajadao | 1.425 | 0.09 |
| Um Pente | 1.392 | 0.07 |
| Vai Caralho | 0.667 | 0.11 |
| Vai malandra | 0.127 | 0.82 |
| Warning | 1.064 | 0.30 |
| Yeah Glo | 0.427 | not captured in the last run (0.96 earlier) |

## What was tried
- The kick vote has little contrast on four-on-the-floor tracks, since kicks hit every beat.
- A second vote from where energy changes at 8-bar boundaries disagreed with the kick vote by a different amount on each track. Inconclusive.
- The heuristic was left unchanged because there is no ground truth.

## Needed
- Ground-truth first downbeats (seconds, or which beat is "1") for at least 3 or 4 tracks, e.g. Smack Yo', Solao, Trajadao, The Weekend.
- Then: measure the current heuristic against them, try alternatives (bar-phase accumulation of low-band onsets, beat_this via `.[beat]`, bass/harmonic change votes), and keep the one that wins.

## Done when
- Detected downbeat is within one beat period of the ground truth on all labeled tracks, and confidence is low only when the answer is actually uncertain.
- A regression test uses the labeled values (small synthetic or short excerpt fixtures).
- `ANALYSIS_VERSION` is bumped and the tracks are re-analysed.

## Related
- `override_track(first_downbeat_sec=...)` is the manual workaround.
- Phrase alignment on Solao → Trajadao is tested against the detected phrase length (32) although Solao's sections look like multiples of 8. Separate, but easier to judge once downbeats are trusted.
