from __future__ import annotations

import librosa
import numpy as np
import soxr

from ..core.units import SR
from .analyze import ASR, HOP, beat_grid, downbeat_estimate, pick_grid
from .analyze import grid_strength as strength_of
from .confidence import attach, combine, unit
from .harmony import harmony_block
from .key import harmonic_neighbours, key_analysis
from .spectral import bar_features, frame_features, loudness_block, spectrum_block, stereo_block, waveform
from .structure import cue_points, structure
from .tension import lead_ins, tension_block
from .vocals import proxy


def _mono(x: np.ndarray) -> np.ndarray:
    return soxr.resample(x.mean(0), SR, ASR).astype(np.float32)


def analyze_audio(audio: np.ndarray, path: str = "", bitrate_kbps: int | None = None, stems: dict[str, np.ndarray] | None = None) -> dict:
    dur = audio.shape[1] / SR
    y = _mono(audio)
    ys = {k: _mono(v) for k, v in (stems or {}).items()}
    g, grid_source = pick_grid(beat_grid(y), ys.get("drums"), dur)
    fd, db_comps = downbeat_estimate([ys[k] for k in ("drums", "bass") if k in ys] or [y], g["t0"], g["period"])
    env = g["env"]
    bpm = 60.0 / g["period"]
    grid_strength = strength_of(g, dur)
    tempo = {
        "bpm": round(bpm, 3), "bpm_rounded": round(bpm), "first_downbeat_sec": round(fd, 4), "beats_per_bar": 4,
        "beat_period_sec": round(g["period"], 6), "grid_stable": bool(g["grid_stable"]),
        "grid_strength": round(grid_strength, 3), "beat_strength": round(unit(grid_strength, 1.0, 3.0), 3),
        "phase_drift_ms": [round(o * 1000) for o in g["offsets"]],
        "alternatives": {"half_time": round(bpm / 2, 2), "double_time": round(bpm * 2, 2)},
        "grid_source": grid_source, "stems_used": bool(ys), "n_beats": int((dur - g["t0"]) // g["period"]),
        "n_bars": int((dur - fd) // (4 * g["period"])),
    }
    offs = np.abs(g["offsets"])
    attach(tempo, {"grid_strength": unit(grid_strength, 1.0, 3.0), "phase_agreement": float((offs < 0.03).mean()) if len(offs) else 0.0,
                   "phase_evidence": len(offs) / 8})
    tempo["downbeat_confidence"], tempo["downbeat_confidence_components"] = combine({"phase_contrast": unit(db_comps["phase_contrast"], 0, 0.25)})
    key = key_analysis(y, dur)
    key["harmonic_neighbours"] = harmonic_neighbours(key["camelot"])
    loud = loudness_block(audio, path, dur)
    spec = spectrum_block(audio.mean(0), bitrate_kbps)
    ff, S, logmel, H = frame_features(y, env)
    bf = bar_features(S, logmel, y, g, fd, dur)
    st = structure(bf, fd, tempo["confidence"])
    cues = cue_points(st, g, fd, dur, bf["n_bars"], tempo["confidence"], tempo["downbeat_confidence"])
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=ASR, hop_length=HOP)
    onset_density = float(onsets.size / dur)
    if st["sections"]:
        on_sec = librosa.frames_to_time(onsets, sr=ASR, hop_length=HOP)
        st["onset_curve"] = [round(float(((on_sec >= fd + k * bf["bar_sec"]) & (on_sec < fd + (k + 1) * bf["bar_sec"])).sum() / bf["bar_sec"]), 2)
                             for k in range(bf["n_bars"])]
        lead_ins(S, st["sections"], fd, bf["bar_sec"], g["period"])
    clar = min(grid_strength / 3.0, 1.0)
    norm = lambda v, a, b: float(np.clip((v - a) / (b - a), 0, 1))
    energy = (0.35 * norm(loud["integrated_lufs"], -20, -6) + 0.25 * norm(onset_density, 1, 8)
              + 0.20 * norm(spec["bass_to_total"], 0.2, 0.7) + 0.20 * norm(ff["spectral_centroid_hz"], 1000, 4000))
    terms = [norm(loud["integrated_lufs"], -20, -6), norm(onset_density, 1, 8), norm(spec["bass_to_total"], 0.2, 0.7),
             norm(ff["spectral_centroid_hz"], 1000, 4000)]
    return {
        "tempo": tempo, "key": key, "loudness": loud, "stereo": stereo_block(audio), "spectrum": {**spec, **ff},
        "energy": attach({"score": round(energy, 3), "level_1_10": int(round(1 + 9 * energy)), "onset_density_per_sec": round(onset_density, 2),
                          "danceability_heuristic": round(float(0.5 * clar + 0.3 * (1.0 if g["grid_stable"] else 0.3)
                                                                + 0.2 * norm(onset_density, 1, 6)), 3),
                          "note": "heuristic composite of loudness, onset density, bass share and brightness"},
                         {"component_agreement": 1.0 - unit(float(np.std(terms)), 0, 0.5),
                          "loudness_finite": float(np.isfinite(loud["integrated_lufs"])),
                          "grid_clarity": clar}),
        "structure": st, "mix_points": cues, "tension": tension_block(bf, st), "harmony": harmony_block(bf, st),
        "vocal_proxy": proxy(S, H, fd, bf["bar_sec"], bf["n_bars"]), "duration_sec": round(dur, 3),
        "_waveform": waveform(audio, int(min(1200, max(dur * 2, 50)))),
    }
