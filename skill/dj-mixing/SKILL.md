---
name: dj-mixing
description: Plan and execute DJ transitions and mixes from track analysis data. Use whenever the task involves mixing, blending or transitioning between songs, choosing a DJ technique, building tension into a drop, beatmatching, harmonic mixing, using stems, or ordering tracks into a DJ set.
---

# DJ mixing

This skill teaches you to think like a DJ when you cannot hear. You perceive music only through analysis data, so your job is to read that data the way a DJ reads a waveform and their own ears, choose a technique that serves the musical goal, express it as a precise plan, render it, and check the measurements of what you rendered.

## Capability names used here

This skill refers to capabilities, not tool names. Map each to whatever your server exposes:

| Capability | What you use it for |
| --- | --- |
| Track analysis | Tempo, beatgrid, downbeats, phrases, sections, key, loudness, energy, bands, vocals, tension features |
| Region query | Detail for a specific range (bars, sections, cues) |
| Pair report / candidates | Compatibility between two tracks and suggested transitions |
| Plan schema | The format your plan must follow, with units and ranges |
| Render | Execute a plan into audio (preview or final) |
| Validation | Measure rendered audio against the plan's intent |

Before your first plan in a session, read the plan schema and check which capabilities are available (stems, key lock, effects, max lanes). Never plan a technique whose prerequisites are unavailable.

## The workflow

Follow these steps for every transition. Skipping step 6 is the most common way to ship a bad mix.

1. **Clarify the goal and the DJ style.** Smooth, energetic, dramatic, emotional, surprising, or experimental? Which style: genre specialist, club, turntablist, mobile/event, radio, or open format? The style sets transition length, how tempo gaps are bridged, and which techniques fit; read `references/dj-styles.md`. If the user didn't say, infer both from context (a wedding → mobile/event, smooth; a techno night at peak time → genre specialist, energetic) and state your assumptions.
2. **Analyze both tracks.** Read the summary first. Request region detail only for the windows you are considering.
3. **Find the windows.** Locate exit regions in track A (outro, breakdown, end of a chorus) and entry regions in track B (intro, build, a section with low vocals). Windows must start on a phrase boundary.
4. **Read the pair.** Tempo ratio, key relation, loudness gap, low-end overlap, vocal overlap. See "Reading the song information" below.
5. **Choose and parameterize a technique.** Use the decision guide below, then the recipe in `references/techniques.md`. Express every timing in bars or beats, not seconds.
6. **Render and validate.** Render a preview of the transition plus 4–8 bars either side. Compare the validation report against your intent.
7. **Iterate.** Adjust one or two parameters at a time based on the numbers. Stop when the measurements match your intent, usually within 2–3 rounds.
8. **Explain.** Tell the user what you did and why, in musical language ("I swapped the basslines on the downbeat of bar 17 so the two kick drums never fight").

## Reading the song information

Treat analysis values as evidence with uncertainty, never as truth.

**Confidence.** Below about 0.7, treat a value as a hint. For low-confidence tempo, check half- and double-time readings. For low-confidence key, prefer techniques that limit harmonic overlap (cuts, short blends, drum-only sections).

**Tempo.** Tempo ratio r = BPM_target / BPM_source. Changing tempo by 3% shifts pitch by about half a semitone if key lock is off; semitone shift = 12 × log2(r). Within ±3%, most music stretches cleanly. From 3% to 6%, prefer key lock and shorter overlaps. Beyond about 8%, don't beatmatch: use a cut, an echo out, a half/double-time relationship, or a tempo ramp over several phrases.

**Beats and phrases.** One bar = 4 × 60 / BPM seconds (1.875 s at 128 BPM). Most dance music phrases in 8, 16 or 32 bars. Transitions that start and end on phrase boundaries sound intentional; transitions that don't sound like mistakes, regardless of how well the beats line up.

**Key.** Use Camelot relationships from `references/music-theory.md`. Same key, ±1 on the wheel, or switching A↔B at the same number are safe for long blends. Two or more steps away: keep harmonic overlap short, use drums-only regions, or make the clash deliberate.

**Energy and sections.** An energy curve tells you where a track breathes. Introduce a track where its energy is low and rising; leave a track where its energy is falling or about to break down. Two high-energy sections overlapping for long usually sounds cluttered.

**Bands.** If both tracks have strong low-frequency energy in the overlap, never leave both basslines at full level: swap or cut one. Mid and high overlap is usually fine.

**Vocals.** Two lead vocals overlapping almost always sounds like a mistake. Let a vocal finish its phrase before the next one starts, or remove one with stems.

**Loudness.** Match the integrated loudness of the two regions to within about 1 LU unless the plan intends a jump. Matching averages across whole tracks is not enough; compare the regions you are mixing.

## Choosing a technique

Start from the situation, not from your favourite technique. Full recipes are in `references/techniques.md`.

| Situation | Good choices |
| --- | --- |
| Tempos within ~3%, compatible keys, both tracks have DJ-friendly intro/outro | Long blend (16–32 bars), bass swap |
| Compatible tempos, keys clash | Short blend on drum-only sections, bass swap with mids cut, phrase cut |
| Tempos far apart (> ~8%) | Echo out into new track, phrase cut, half/double-time mix, tempo ramp over several phrases, brake or spinback |
| Outgoing vocal still going at the exit point | Wait for the phrase end, loop the instrumental bar, or remove the vocal stem |
| Want an energy lift | Bass swap on a phrase boundary, filter build into the incoming drop, key lift (+1 or +2 semitones) |
| Want a dramatic moment | Build with roll + HPF + riser, pre-drop gap, drop into track B |
| Want emotional depth | Resolve the build into a breakdown, relative major/minor move, acapella over a sparse instrumental |
| Switching genre or tempo family | Echo out, reverb wash, silence break, or a cut on the downbeat |
| Both tracks have strong drops at the same tempo and compatible keys | Double drop (advanced; validate low end carefully) |
| Track B has no usable intro | Loop extension of its first clean bar, or cut into it |
| Track A has no usable outro | Loop extension of its last clean phrase, or echo out |

Filter this table through the DJ style: a genre specialist rarely cuts, a radio DJ rarely beatmatches, an open-format DJ rarely runs a 32-bar blend. When several fit, prefer the one that best matches the goal, then the one with fewer prerequisites. Use candidate scores as a palette, not a verdict: a lower-scored candidate may be the right artistic choice.

## Building tension and emotion

When the goal includes tension, drama or emotion, read `references/tension.md`. In short:

- Tension rises with loudness, brightness, rhythmic density, pitch height and dissonance.
- Build on an accelerating curve u^k (k ≈ 1.5 patient, k ≈ 4 holds back then surges), never linearly.
- Sweep filters on a log scale; halve roll lengths in steps; add a short gap before the drop.
- The emotion comes from what the tension resolves into: a drop (euphoric), a relative-key shift (uplift or bittersweet), a breakdown (intimate), or no release at all (unsettling).

## Rules of thumb, and when to break them

Follow these by default. Break any of them deliberately when the goal calls for it, and say so in the plan's intent so validation treats it as intended.

1. Start and end transitions on phrase boundaries.
2. Never let two full basslines play together for more than a beat.
3. Never overlap two lead vocals.
4. Keep tempo changes within your quality limit (about ±6% with key lock).
5. Match loudness across the transition.
6. Keep harmonic overlap short when keys clash.
7. Leave headroom: the master should never clip.

Deliberate breaks that work: a vocal clash for one beat as a call-and-response; a key clash during a build to raise tension; a loudness spike on a drop; a hard cut mid-phrase for shock value. Accidental breaks never work.

## Reading validation

| Measurement | Acceptable by default | If it fails |
| --- | --- | --- |
| Beat drift | < 10 ms average, < 20 ms max | Re-check downbeat alignment; shorten the overlap; check for a dynamic beatgrid |
| Loudness jump | < 1 LU unless intended | Adjust the incoming track's gain |
| True peak | ≤ −1 dBTP | Lower lane gain, not just the master limiter |
| Low-end overlap | < 1 beat with both basses full | Move the bass swap point or cut one low band earlier |
| Vocal collision | 0 unless intended | Move the entry to the next phrase or mute a vocal stem |
| Unplanned silence | 0 | Check region boundaries and loop points |

A plan that passes validation can still sound wrong, and validation numbers are proxies. When you're unsure, render two variations and compare their reports, and tell the user which one you'd pick and why.

## Reference files

- `references/dj-styles.md` — the main DJ styles (genre specialist, club, turntablist, mobile/event, radio, open format), what each values, its signature techniques, and BPM-bridging strategies for large tempo gaps. Read it at the start of every task.
- `references/techniques.md` — recipes for each technique: when to use it, prerequisites, step-by-step envelopes, defaults and failure modes. Read the entry for any technique before planning it.
- `references/music-theory.md` — Camelot wheel, key relationships, tempo maths, phrase lengths and typical BPM by genre.
- `references/tension.md` — the maths of tension, release and emotional colour, with a worked example.
- `references/set-building.md` — ordering tracks into a set: energy arcs, tempo arcs, key journeys and constraints.
