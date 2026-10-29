# Set building

A set is a journey, not a playlist. Plan the shape of the whole set first, then pick tracks and transitions that serve it.

## Workflow

1. **Define the set.** Duration, context (warm-up, peak time, closing, background, livestream), audience energy, genre range, fixed tracks (opener, closer, must-plays).
2. **Analyze the pool.** Tempo, key, energy, sections and loudness for every candidate track.
3. **Draw the arcs.** Choose target curves for energy, tempo and (optionally) mood across the set.
4. **Build the compatibility graph.** For every pair, note tempo ratio, key move, energy difference and any interesting contrast.
5. **Order the tracks.** Find a path through the graph that follows the arcs and respects constraints.
6. **Trim.** Decide which part of each track plays and for how long (most tracks play 3–5 minutes of a 6–8 minute club version).
7. **Plan each transition** with the main workflow in SKILL.md.
8. **Preview the whole set** at low quality, then validate each transition.

## Energy arcs

Rate each track's energy 1–10 from analysis (loudness, density, brightness, low-end strength). Then pick an arc:

| Context | Arc |
| --- | --- |
| Warm-up | Start 3–4, rise slowly to 6; avoid big drops |
| Peak time | Waves between 7 and 10, with short dips to 6 every 3–5 tracks |
| Closing | Peak, then gradual release to 4–5; end on something memorable |
| Background / lounge | Flat 3–5, smooth transitions only |
| Journey (2+ hours) | Several waves, each peak slightly higher, a clear climax, then release |

Rules: raise energy by at most about 2 points per transition unless you want a jolt; after 3–4 tracks at the peak, give the audience a dip so the next peak lands.

## Tempo arcs

- Change tempo gradually: about 1–3 BPM per transition, or move in steps at phrase-cut transitions.
- Use tempo-family jumps (e.g. 87 → 174 via half/double time) for genre changes.
- Peak time often runs a little faster than warm-up (e.g. 120 → 126 in house).

## Key journeys

- Default: move ±1 on the Camelot wheel or switch letters at the same number.
- Use a +2 or +7 key lift at the start of a new energy wave.
- Avoid staying in one key for more than 3–4 tracks; it flattens the set.
- A deliberate clash is fine at a cut, especially at a genre or mood change.

## Ordering heuristics

When the pool is large, score each possible next track by:

    next_score = w_energy · energy_fit + w_tempo · tempo_fit + w_key · key_fit + w_contrast · contrast + w_freshness · freshness

- energy_fit: how close the track's energy is to the arc's target at that point.
- tempo_fit: how small the tempo change is (within the allowed limit).
- key_fit: from the key-move table in music-theory.md.
- contrast: reward an interesting difference occasionally (new texture, genre turn) so the set doesn't feel like one long track.
- freshness: penalize repeats of artist, label, or sound within the last N tracks.

A greedy choice (best next track each time) works for small pools. For larger pools, search several orderings (beam search) and keep the one with the best total score along the arc.

## Constraints to always respect

- Fixed opener and closer stay in place.
- Must-include tracks are included.
- Total duration within the target (±5% by default).
- No artist repeated within N tracks (default 5) unless the user wants it.
- No transition exceeds the tempo-change limit unless it uses a cut, echo out or ramp.

## Recurring motifs and callbacks

A strong set references itself: bring back an earlier track's hook as an acapella overlay later, or close with a track that echoes the opener's key or melody. Note candidate motifs (hooks, vocal tags, distinctive sounds) during analysis so you can plan callbacks.

## Explaining the set

When you deliver a set, give the user a short tracklist with timestamps and one line per transition: the technique and why. Point out the main energy peaks and the key moments, so they understand the journey without listening yet.
