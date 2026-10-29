# Music theory for mixing

## Camelot wheel

Minor keys are A, major keys are B. Each number pair is a relative major/minor (same notes).

| Camelot | Minor (A) | Camelot | Major (B) |
| --- | --- | --- | --- |
| 1A | A♭ / G♯ minor | 1B | B major |
| 2A | E♭ / D♯ minor | 2B | F♯ / G♭ major |
| 3A | B♭ minor | 3B | D♭ major |
| 4A | F minor | 4B | A♭ major |
| 5A | C minor | 5B | E♭ major |
| 6A | G minor | 6B | B♭ major |
| 7A | D minor | 7B | F major |
| 8A | A minor | 8B | C major |
| 9A | E minor | 9B | G major |
| 10A | B minor | 10B | D major |
| 11A | F♯ minor | 11B | A major |
| 12A | D♭ / C♯ minor | 12B | E major |

## Key moves (from any key N)

| Move | Example from 8A | Effect | Use for |
| --- | --- | --- | --- |
| Same key | 8A → 8A | Seamless | Long blends, layering |
| ±1, same letter | 8A → 9A or 7A | Smooth; +1 slightly brighter | Long blends |
| Same number, other letter | 8A → 8B | Relative major/minor: mood shift, no clash | Emotional lift (A→B) or bittersweet turn (B→A) |
| +2, same letter | 8A → 10A | Up a whole tone: energy boost | Phrase cuts, after breakdowns |
| +7, same letter | 8A → 3A | Up a semitone: classic key lift | Short blends, cuts at peaks |
| ±1 and switch letter | 8A → 9B | Diagonal: moderate change | Short blends |
| Anything else | | Likely clash | Cuts, drum-only overlaps, or deliberate tension |

**Pitch shifting and the wheel:** shifting a track up 1 semitone moves it +7 on the wheel (mod 12); up 2 semitones moves it +2. Use this to make two incompatible tracks compatible, staying within about ±2 semitones to avoid artefacts.

**Local harmony matters more than global key.** A track labelled 8A may have a drum-only intro with no key at all, which can blend with anything. Check the chroma of the actual regions you overlap.

## Tempo maths

- Beat length = 60 / BPM seconds. Bar (4/4) = 240 / BPM seconds.
- Tempo ratio r = BPM_target / BPM_source; percentage change = (r − 1) × 100.
- Pitch change without key lock = 12 × log2(r) semitones. 3% ≈ 0.51 semitone; 6% ≈ 1 semitone.
- 16 bars at 128 BPM = 30 s; at 174 BPM = 22 s; at 90 BPM = 42.7 s.
- Half/double-time: if BPM_A ≈ 2 × BPM_B (within ~3%), they can be beatmatched at a 2:1 ratio.

| Tempo change | Typical quality |
| --- | --- |
| ≤ 3% | Clean with or without key lock |
| 3–6% | Fine with key lock; pitch shift audible without it |
| 6–10% | Artefacts likely, especially on vocals and sustained notes |
| > 10% | Avoid beatmatching; use cuts, effects exits or ramps |

## Phrase lengths and typical tempos by genre

Ranges are typical, not rules; always trust the analysis over this table.

| Genre | Typical BPM | Phrase habits | Mixing style |
| --- | --- | --- | --- |
| House / deep house | 118–128 | 16/32-bar phrases, long DJ intros/outros | Long blends, bass swaps |
| Tech house | 124–128 | 16/32 bars | Long blends, loops, filter builds |
| Techno | 125–140 | 16/32/64 bars, often minimal harmony | Very long blends, layering |
| Trance | 130–140 | 16/32 bars, long breakdowns | Blends, breakdown transitions, key moves |
| Drum & bass | 170–176 | 16/32 bars, drop after ~32-bar intro | Double drops, fast bass swaps |
| Dubstep | 140 (half-time feel) | 16 bars, drop-centred | Drop swaps, double drops, cuts |
| Hip-hop | 85–100 | 4/8-bar verses, 8-bar hooks | Cuts, echo outs, scratches, short blends |
| Pop / open format | 90–130 | Verse/chorus 8 bars, short intros | Cuts, short blends, echo outs |
| Afrobeats / amapiano | 100–115 | 8/16 bars | Blends, percussion layering |
| Reggaeton / dancehall | 88–100 | 8 bars | Cuts, short blends |
| Disco / funk | 110–125 | Live drummers: drift likely | Dynamic beatgrid; short blends |

Live-drummed music (disco, funk, rock, older soul) drifts. Use a dynamic beatgrid, keep overlaps short, and validate drift across the whole overlap.

## Frequency bands

| Band | Range | What lives there | Mixing note |
| --- | --- | --- | --- |
| Low | ~20–150 Hz | Kick, sub, bassline | Only one track at full level at a time |
| Mid | ~150 Hz–2 kHz | Vocals, chords, leads, snare body | Main source of harmonic and vocal clash |
| High | ~2–20 kHz | Hats, cymbals, air, sibilance | Usually safe to overlap; brightness adds energy |
