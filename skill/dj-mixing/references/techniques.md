# Technique recipes

Each recipe lists when to use the technique, what it needs, how to build it from primitives, sensible defaults and what to check in validation. Timings are in bars relative to the start of the transition (bar 0), with track A outgoing and track B incoming, unless stated otherwise. Every start and end point should sit on a downbeat, and usually on a phrase boundary.

Primitives referred to below: gain envelope, EQ band envelope, filter envelope (LPF/HPF with resonance), rate curve, loop, roll, delay, reverb, gate, reverse, stem level, sample trigger.

## Contents

1. Blends: long blend, short blend, intro–outro blend, bass swap, EQ swap, breakdown transition
2. Cuts: phrase cut, beat cut, drop swap, silence break
3. Effect exits: echo out, reverb out, filter out, brake, spinback, tape stop
4. Builds: filter build, roll build, riser build, outro-to-drop
5. Tempo and key: tempo ramp, half/double-time mix, key lift / key drop, relative-key transition
6. Loops: loop extension, loop roll, back-to-back phrase mix
7. Stems and layering: stem swap, acapella overlay, instrumental bridge, double drop, mashup
8. Performance gestures: scratch, beat juggle, slicer, gate/transformer

---

## 1. Blends

### Long blend
- **Use when:** tempos within ~3%, compatible keys, A has a long outro and B a long intro, goal is smooth or hypnotic.
- **Avoid when:** keys clash, both regions are dense or vocal-heavy.
- **Needs:** beatgrid, phrases.
- **Recipe (16 or 32 bars):**
  1. Sync B to A's tempo (or meet in the middle). Align B's phrase start with A's phrase start.
  2. Bar 0: B enters with low EQ at −∞ (kill) and gain at −6 dB, rising to 0 dB by bar 4.
  3. Bars 0–8: B's highs and mids come up; A unchanged.
  4. Bar 8 (or 16 in a 32-bar blend): bass swap — A's low to kill, B's low to 0 dB within 1 beat on the downbeat.
  5. Remaining bars: A's mids and highs fade out on an equal-power curve.
- **Defaults:** 16 bars; equal-power curves; bass swap at the midpoint.
- **Watch:** beat drift across the whole overlap; low-end overlap; loudness continuity.

### Short blend
- **Use when:** tempos compatible but keys or textures clash, or the set needs pace.
- **Recipe (4–8 bars):** B enters with low cut; bass swap on the first downbeat of the second half; A fades out by the end.
- **Defaults:** 8 bars.
- **Watch:** harmonic clash (keep it short); vocal collision.

### Intro–outro blend
- **Use when:** A has a beat-only outro and B a beat-only intro (common in club edits).
- **Recipe:** a long blend that spans exactly the overlap of A's outro and B's intro; length = min(outro bars, intro bars), rounded down to 8 or 16.
- **Watch:** that the overlap ends before either track's melodic content begins, unless keys are compatible.

### Bass swap
- **Use when:** almost any beatmatched transition; essential whenever both tracks have strong bass.
- **Recipe:** both tracks play with B's low band killed. On a phrase-boundary downbeat, A's low goes to kill and B's low to 0 dB simultaneously, within 1 beat (or instantly).
- **Defaults:** instant swap on the downbeat.
- **Variations:** swap on the first beat of a drop for maximum lift; swap gradually over 1 bar for a softer feel.
- **Watch:** low-end overlap should be under one beat.

### EQ swap
- **Use when:** you want each frequency band to hand over separately, which creates a sense of the new track taking over gradually.
- **Recipe:** hand over highs first (bar 0–4), then mids (bar 4–8), then lows (bar 8, on the downbeat).
- **Watch:** loudness dip in the middle; correct with gain if needed.

### Breakdown transition
- **Use when:** A reaches a breakdown (low drums, sparse) and B can enter quietly; good for genre changes and emotional moments.
- **Recipe:** start B's intro or breakdown during A's breakdown; let A end or fade inside its breakdown; B's next build carries the energy forward.
- **Watch:** keys during the overlap (breakdowns are usually melodic, so harmony matters more).

---

## 2. Cuts

### Phrase cut
- **Use when:** tempos or keys are far apart, the energy should change instantly, or the genre is cut-friendly (hip-hop, pop, open format).
- **Needs:** downbeats and phrases.
- **Recipe:** A plays to the last beat of a phrase; B starts on its first downbeat at exactly the next beat. Optionally fade A's last 1/16 beat to avoid a click.
- **Watch:** loudness jump; unplanned silence; clicks.

### Beat cut
- Same as a phrase cut but on any downbeat. Use for quick back-and-forth cuts or call-and-response. Sounds intentional only if repeated or clearly rhythmic.

### Drop swap
- **Use when:** both tracks have drops at compatible tempos; high-energy peak-time moment.
- **Recipe:** play A's build; at the moment A's drop would hit, cut to B's drop instead. Optionally keep A's build effects (riser, roll) running for 1 beat into B.
- **Watch:** loudness jump (usually wanted, but cap it); beat alignment at the swap.

### Silence break
- **Use when:** you want maximum surprise, a genre reset, or a pre-drop gap.
- **Recipe:** A stops dead on a downbeat; silence for 1 beat to 1 bar; B starts on the next downbeat.
- **Watch:** declare the silence as intended so validation doesn't fail it.

---

## 3. Effect exits

### Echo out
- **Use when:** A must leave cleanly but B can't be beatmatched (tempo or genre change), or you want a classic dramatic exit.
- **Recipe:**
  1. On the last beat of a phrase in A, engage a beat-synced delay (1/2 or 3/4 beat) with feedback 50–70%, wet 100%.
  2. Cut A's dry signal at the same moment; let the delay tail ring.
  3. B starts on the downbeat 1–4 beats later, at its own tempo.
  4. Fade the echo tail over 1–2 bars, with an HPF rising so tails don't muddy B's low end.
- **Defaults:** 1/2-beat delay, 60% feedback, 2-bar tail.
- **Variations:** instrumental echo out (echo only the vocal stem); echo freeze (feedback 100% for 1 bar, then cut).
- **Watch:** low-end overlap from the tail; peak level.

### Reverb out
- Like echo out but with a long reverb (decay 4–8 s) and pre-cut HPF. Feels spacious and emotional. Works well into ambient or slow tracks.

### Filter out
- **Recipe:** over 4–8 bars, sweep A's HPF from 30 Hz up to 1–3 kHz on a log curve (or LPF down to 200–400 Hz for a muffled exit) while B rises. Add resonance for a sharper effect.

### Brake
- **Recipe:** A's rate falls from 1.0 to 0 over 1/2 to 2 beats (exponential curve). B starts on the next downbeat.
- **Use for:** hip-hop, open format, comedic or dramatic stops.

### Spinback
- **Recipe:** A's rate jumps to −3 to −5 (fast reverse) and decays to 0 over 1–2 beats. Optional reverb tail. B starts on the next downbeat.

### Tape stop
- Like a brake but slower (1–2 bars) with a pitch drop and lo-fi character. Works into slower tracks.

---

## 4. Builds

### Filter build
- **Use when:** you want to create tension into B's drop or into a new phrase.
- **Recipe (8 bars):** A's HPF sweeps 30 Hz → 2 kHz on f_c(u) = f0 · (f1/f0)^(u^k), k ≈ 2; resonance rising from 0 to moderate. On B's downbeat, A is cut and B enters at full range.
- **Watch:** tension curve match; loudness jump at the drop (usually intended).

### Roll build
- **Recipe:** over the last 8 bars of A, roll the current beat with length halving every 2 bars: 1 → 1/2 → 1/4 → 1/8 beat. Combine with a filter build. Cut on the final downbeat.
- **Watch:** that roll starts on a beat boundary; no clicks at roll edges.

### Riser build
- **Recipe:** trigger a noise or synthesized riser rising 12 semitones over 4–8 bars, gain rising on u^k. Cut it exactly on the drop downbeat, or 1 beat earlier with a gap.

### Outro-to-drop
- **Use when:** A ends with a fading outro and B has a strong drop.
- **Recipe:** let A's outro play; start B's build so its drop lands on the first downbeat after A's outro ends (count back from B's drop to find B's start time). Kill B's low until the drop.
- **Watch:** beat alignment where B's build overlaps A's outro.

---

## 5. Tempo and key

### Tempo ramp
- **Use when:** tempo gap is more than ~6% and you want continuity rather than a cut.
- **Recipe:** over 16–64 bars before the transition, ramp A's tempo toward B's on a smooth curve. Keep each 8-bar step under ~2% change. Key lock on if stretch exceeds ~3%.
- **Watch:** artefacts at extreme stretch; drift during the ramp.

### Half/double-time mix
- **Use when:** one track is close to half or double the other (e.g. 70 and 140, 87 and 174).
- **Recipe:** sync B at the 2:1 or 1:2 ratio; align B's downbeat to every second (or every) downbeat of A. Then use a bass swap or phrase cut.
- **Watch:** beat alignment at the ratio; feel (half-time drops feel heavier).

### Key lift / key drop
- **Use when:** you want a gear change in energy (up) or mood (down).
- **Recipe:** move to a track 1 or 2 semitones higher (Camelot +7 or +2) on a phrase cut or after a breakdown, or pitch-shift B by +1/+2 semitones with key lock. Key drops do the reverse and feel darker.
- **Watch:** stretch artefacts if pitch-shifting more than ~2 semitones.

### Relative-key transition
- **Use when:** you want emotional colour without harmonic clash: minor to relative major feels like uplift; major to relative minor feels bittersweet.
- **Recipe:** pair tracks at the same Camelot number with different letters (8A ↔ 8B); use a long blend or a breakdown transition so the harmony change is heard.

---

## 6. Loops

### Loop extension
- **Use when:** A's outro or B's intro is too short for the blend you want, or you need more time.
- **Recipe:** loop a clean 4- or 8-bar region with no vocal or melodic ending; repeat it N times; exit on the loop's last downbeat.
- **Watch:** loop points on exact downbeats; no clicks; listeners notice exact repetition after about 16 bars, so vary it with filters or EQ.

### Loop roll
- **Recipe:** a short roll (1/4 to 1 beat) with slip behaviour: when it ends, the track resumes where it would have been. Use as a fill at the end of a phrase.

### Back-to-back phrase mix
- **Use when:** you want to tease B before committing, or create a call-and-response between two tracks.
- **Recipe:** alternate A and B every 4 or 8 bars (cut on downbeats), then commit to B.
- **Needs:** tight beatmatch; similar energy.

---

## 7. Stems and layering

All of these need stem separation. Check the separation quality report first; avoid stems with heavy bleed.

### Stem swap
- **Recipe:** keep A's vocals and melody; replace A's drums and/or bass with B's (key- and tempo-matched) for 8–16 bars; then bring in B's remaining stems and fade A out.
- **Use when:** you want a seamless, almost remix-like transition between tracks of similar style.

### Acapella overlay
- **Recipe:** place A's or a third track's vocal stem over B's instrumental (B with its vocal stem muted). Key and tempo must be compatible; align vocal phrases to B's phrase boundaries.
- **Watch:** harmonic clash between vocal and instrumental; vocal collision with B's vocal.

### Instrumental bridge
- **Recipe:** mute A's vocal stem 4–8 bars before the exit, so the transition happens between two instrumentals; bring B's vocal in after the transition.
- **Use when:** vocals would otherwise collide.

### Double drop
- **Use when:** two drops at the same tempo with compatible keys; peak-time, high-energy moment.
- **Recipe:** align both tracks so their drops land on the same downbeat. Give the low end to only one (kill the other's low or bass stem). Optionally keep both drums for impact.
- **Watch:** low-end overlap (critical); peak level; harmonic clash.

### Mashup
- **Recipe:** combine stems from two or more tracks for an extended section: e.g. vocal from A, drums from B, melody from C. Treat it as a mini-arrangement with its own phrase plan.
- **Watch:** every collision check; consider rights for any distributed output.

---

## 8. Performance gestures

These are live hand techniques, rendered here as rate and gate curves. Use them sparingly; they read as showmanship.

### Scratch
- **Recipe:** over 1–2 beats, a rate curve oscillating between forward and backward (e.g. +1 → −1 → +1), with an optional gate pattern on the fader. Common shapes: baby scratch (rate only), chirp (gate closes on the backstroke), transformer (rapid gate on a forward stroke).

### Beat juggle
- **Recipe:** two copies of the same break on two lanes, alternating every beat or half-beat to create a new pattern. Needs a tight beatgrid.

### Slicer
- **Recipe:** divide a 1-bar region into 8 or 16 slices and re-sequence them (e.g. 1-1-3-1-5-5-7-8) for 1–4 bars, then resume.

### Gate / transformer
- **Recipe:** gate the lane on a rhythmic pattern (1/8 or 1/16 notes) with fast attack. Use on a vocal or sustained sound during a build or as a fill.
