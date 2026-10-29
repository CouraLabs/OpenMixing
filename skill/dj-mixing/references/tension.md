# Tension, release and emotion

Emotion in a transition comes from the shape of the tension curve and what it resolves into. This follows expectation theory in music cognition: listeners feel the most when a prediction is held in suspense and then resolved. These are proxies grounded in research, not a formula for feelings; always validate and, where possible, compare variations.

## 1. Tension index

Tension is a weighted sum of normalized (0–1) features measured over time:

    T(t) = Σ w_i · f_i(t)

| Feature | Raises tension when | Primitive that drives it |
| --- | --- | --- |
| Loudness | Rising | Gain envelope |
| Brightness (spectral centroid) | Rising | HPF sweep, riser, noise |
| Rhythmic density (onsets/s) | Rising | Roll, beat repeat, added percussion |
| Pitch height | Rising | Riser, pitch effect curve, key lift |
| Dissonance (roughness) | Rising | Harmonic overlap of two tracks |
| Low-end absence | Bass removed | HPF, bass EQ kill, bass stem mute |

Default weights: equal. Shift weight toward the features the genre relies on (brightness and density in EDM; harmony and pitch in pop and trance).

## 2. Build shape

With u = t / D running from 0 to 1 across a build of length D, use an accelerating curve:

    T(u) = u^k,  k > 1

| k | Feel |
| --- | --- |
| 1.0 | Linear: mechanical, use rarely |
| 1.5–2 | Patient, steady build |
| 3–4 | Holds back, then surges at the end |

Apply the same u^k to every envelope in the build so all features rise together.

**Filter sweep (log-scale, because pitch perception is logarithmic):**

    f_c(u) = f0 · (f1 / f0)^(u^k)

Typical HPF: f0 = 30 Hz, f1 = 1–3 kHz. Typical LPF exit: f0 = 20 kHz, f1 = 200–400 Hz.

**Roll subdivision:** repeat length L_n = beat / 2^n. At 128 BPM: 469 → 234 → 117 → 59 ms. Halve every 2 bars over the last 8. At the final step the repeats occur ~17 times per second and start to merge into a tone, which signals an imminent break.

**Riser:** pitch f(u) = f0 · 2^(s · u^k / 12), where s is the total rise in semitones (12 = one octave).

## 3. Harmonic tension

Roughness between two partials with frequencies f_i, f_j and amplitudes a_i, a_j (Plomp–Levelt, Sethares parameters):

    R = Σ a_i · a_j · (e^(−3.5 · s · Δf) − e^(−5.75 · s · Δf)),  s = 0.24 / (0.0207 · f_min + 18.96)

Use it deliberately: let B's harmony bleed in during A's build so roughness rises, then cut to one consonant harmony at the drop. The collapse of roughness is a large part of what feels like relief. Keep deliberate dissonance to the build only.

## 4. The gap

Right before the drop, remove nearly everything for 1/2 to 1 bar (or leave only a riser tail or a vocal shout). The listener's expectation of the drop is briefly delayed, so it peaks. Declare the gap as intended silence in the plan.

## 5. Release

Impact scales with how far tension falls at the downbeat:

    Impact ∝ T_peak − T_after

At the release, reset every build envelope at once: filter fully open, roll off, riser cut, bass back, harmony consonant.

## 6. Emotional colour

Place the transition on the arousal–valence plane:

- **Arousal (energy):** loudness, brightness, density, tempo.
- **Valence (positive vs. negative feeling):** major vs. minor mode, consonance, brightness.

Plan a trajectory, not a level:

| Build in | Resolve into | Typical feeling |
| --- | --- | --- |
| Minor | Relative major drop (e.g. 8A → 8B) | Euphoric, uplifting |
| Major | Minor drop | Powerful, darker, bittersweet |
| Any | Sparse breakdown (no drop) | Intimate, emotional, space to breathe |
| Any | Key lift (+1 or +2 semitones) drop | Fresh lift, "gear change" |
| Any | No resolution (cut to silence, or into an unrelated track) | Unsettling, surprising |

## 7. Worked example: 8 bars at 128 BPM (15 s), k = 3

The build runs over bars 1–7 (D = 7 bars) and bar 8 holds the gap. Values are at the end of each span.

| Bars | HPF cutoff | Roll | Riser | Harmony |
| --- | --- | --- | --- | --- |
| 1–2 | 30 → 33 Hz | 1 beat | +0.3 st | A only |
| 3–4 | → 66 Hz | 1/2 beat | +2.2 st | B's chords bleed in |
| 5–6 | → 420 Hz | 1/4 beat | +7.6 st | A + B, roughest |
| 7 | → 2 kHz | 1/8 beat | +12 st | Peak |
| 8 | gap (1/2 bar) | off | cut | — |
| Drop | open | off | — | B alone, resolved |

Values follow u^3: the curve stays low for most of the build and surges in the last quarter, which is the point.

## 8. Validation for tension

- Tension curve match: compare measured T(t) to the planned one; correlation should be high (above about 0.8).
- Loudness jump at the drop: should exist if planned, but stay under ~6 LU.
- Intended silence: the gap must be present and the right length.
- Roughness: high during the build, low after the drop.
