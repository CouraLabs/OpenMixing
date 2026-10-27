"""Deep audio analysis: everything computable from the waveform alone.

Output groups: tempo/grid, key, loudness, spectrum, stereo, quality, energy, structure (sections), cue / mix
points, loop candidates and a 3-band waveform overview. Values labelled "heuristic" are engineered scores, not
trained-model outputs.
"""
from __future__ import annotations

import librosa
import numpy as np
import pyloudnorm
import soxr
from scipy.signal import welch

from ..dsp.eq import split_bands
from ..models import SR
from .analyze import ASR, HOP, TARGET_LUFS, beat_grid, downbeat_estimate
from .confidence import attach, combine, unit

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]  # by tonic pitch class
PROFILES = {  # (major, minor)
    "krumhansl": ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "temperley": ([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                  [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
}
BANDS = {"sub": (20, 60), "bass": (60, 250), "low_mid": (250, 500), "mid": (500, 2000), "high_mid": (2000, 6000),
         "high": (6000, 20000)}


def camelot(pc: int, minor: bool) -> str:
    return f"{CAMELOT_MAJOR[(pc + 3) % 12] if minor else CAMELOT_MAJOR[pc]}{'A' if minor else 'B'}"


def key_name(pc: int, minor: bool) -> str:
    return f"{NOTES[pc]} {'minor' if minor else 'major'}"


def rank_keys(chroma_mean: np.ndarray) -> list[dict]:
    res = []
    for pc in range(12):
        for minor in (False, True):
            r = np.mean([np.corrcoef(np.roll(p[int(minor)], pc), chroma_mean)[0, 1] for p in PROFILES.values()])
            res.append({"key": key_name(pc, minor), "camelot": camelot(pc, minor), "score": round(float(r), 4),
                        "_pc": pc, "_minor": minor})
    return sorted(res, key=lambda d: -d["score"])


def key_analysis(y: np.ndarray, dur: float) -> dict:
    yh = librosa.effects.harmonic(y, margin=3)
    tuning = float(librosa.estimate_tuning(y=yh, sr=ASR))
    chroma = librosa.feature.chroma_cqt(y=yh, sr=ASR, tuning=tuning, hop_length=2048)
    ranked = rank_keys(chroma.mean(1))
    best, second = ranked[0], ranked[1]
    margin = best["score"] - second["score"]
    out = {
        "key": best["key"], "camelot": best["camelot"], "margin": round(margin, 4),
        "correlation": best["score"], "tuning_cents": round(tuning * 100, 1),
        "candidates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in ranked[:4]],
        "chroma_mean": [round(float(v), 4) for v in chroma.mean(1) / max(chroma.mean(1).max(), 1e-9)],
    }
    # key over time: detect modulations / mixed-key tracks
    n = chroma.shape[1]
    seg = max(n // 6, 1)
    track = []
    for i in range(0, n - seg + 1, seg):
        r = rank_keys(chroma[:, i : i + seg].mean(1))[0]
        track.append({"start_sec": round(i * 2048 / ASR, 1), "key": r["key"], "camelot": r["camelot"],
                      "score": r["score"]})
    out["key_over_time"] = track
    out["key_stable"] = len({t["camelot"] for t in track}) <= 2
    agree = float(np.mean([t["camelot"] == best["camelot"] for t in track])) if track else 0.0
    return attach(out, {"margin": unit(margin, 0, 0.125), "correlation": unit(best["score"], 0.4, 0.9), "agreement_over_time": agree})


def harmonic_neighbours(cam: str) -> dict:
    """Camelot-wheel compatible keys: same, +/-1 (energy), relative major/minor, +2 (energy boost)."""
    n, m = int(cam[:-1]), cam[-1]
    wrap = lambda k: (k - 1) % 12 + 1
    other = "A" if m == "B" else "B"
    return {"perfect": [cam], "smooth": [f"{wrap(n + 1)}{m}", f"{wrap(n - 1)}{m}"], "relative": [f"{n}{other}"],
            "energy_boost": [f"{wrap(n + 2)}{m}", f"{wrap(n + 7)}{m}"]}


def _db(x, floor=1e-10):
    return 10 * np.log10(np.maximum(x, floor))


def loudness_block(audio: np.ndarray, path: str, dur: float | None = None) -> dict:
    from ..render import measure

    mono = audio.mean(0)
    lufs = pyloudnorm.Meter(SR).integrated_loudness(audio.T)
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((mono ** 2).mean()))
    tp = measure(path) if path else {}
    gain = float(np.clip(TARGET_LUFS - lufs, -12, 12)) if np.isfinite(lufs) else 0.0
    # leading / trailing silence at -50 dBFS (10 ms frames)
    fr = np.sqrt(np.convolve(mono ** 2, np.ones(480) / 480, "valid")[::480])
    loud = np.flatnonzero(_db(fr ** 2) > -50)
    lead = float(loud[0] * 0.01) if len(loud) else 0.0
    trail = float((len(fr) - 1 - loud[-1]) * 0.01) if len(loud) else 0.0
    dur = audio.shape[1] / SR if dur is None else dur
    measured = [tp.get("loudness_range_lu"), tp.get("true_peak_dbtp")]
    return attach({
        "integrated_lufs": round(float(lufs), 2), "loudness_range_lu": tp.get("loudness_range_lu"),
        "true_peak_dbtp": tp.get("true_peak_dbtp"), "sample_peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 2), "crest_factor_db": round(20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9)), 2),
        "clipped_samples": int((np.abs(audio) >= 0.999).sum()), "dc_offset": round(float(mono.mean()), 6),
        "suggested_gain_db": round(gain, 2), "target_lufs": TARGET_LUFS,
        "leading_silence_sec": round(lead, 2), "trailing_silence_sec": round(trail, 2),
    }, {"finite": float(np.isfinite(lufs)), "duration": unit(dur, 0, 30), "ebur128_measured": sum(m is not None for m in measured) / 2,
        "clipping_free": 1.0 - unit(float((np.abs(audio) >= 0.999).sum()), 0, 0.001 * audio.size)})


def stereo_block(audio: np.ndarray) -> dict:
    l, r = audio[0, ::4], audio[1, ::4]
    m, s = (l + r) / 2, (l - r) / 2
    corr = float(np.corrcoef(l, r)[0, 1]) if l.std() > 0 and r.std() > 0 else 1.0
    return {"correlation": round(corr, 4), "width": round(float(np.sqrt((s ** 2).mean() / max((m ** 2).mean(), 1e-12))), 4),
            "is_mono": bool(np.abs(audio[0] - audio[1]).max() < 1e-4)}


def spectrum_block(mono: np.ndarray, bitrate_kbps: int | None) -> dict:
    f, p = welch(mono, SR, nperseg=8192)
    tot = p.sum()
    shares = {k: round(float(p[(f >= lo) & (f < hi)].sum() / tot), 4) for k, (lo, hi) in BANDS.items()}
    pdb = _db(p)
    above = np.flatnonzero(pdb > pdb.max() - 90)
    cutoff = float(f[above[-1]]) if len(above) else 0.0
    quality = None
    if cutoff < 16500 and (bitrate_kbps or 0) >= 256:
        quality = "bitrate is high but content is band-limited below 16.5 kHz: likely transcoded from a lower-quality source"
    elif cutoff < 16500:
        quality = "band-limited below 16.5 kHz (typical of low-bitrate lossy encodes)"
    return {"band_energy_share": shares, "band_energy_db": {k: round(float(_db(v)), 1) for k, v in shares.items()},
            "bass_to_total": round(shares["sub"] + shares["bass"], 4), "content_cutoff_hz": round(cutoff),
            "quality_note": quality}


def frame_features(y: np.ndarray, env: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 1
    H, P = librosa.decompose.hpss(S, margin=2.0)
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=ASR, n_mels=64)
    logmel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13)
    f = {
        "spectral_centroid_hz": float(librosa.feature.spectral_centroid(S=S, sr=ASR).mean()),
        "spectral_bandwidth_hz": float(librosa.feature.spectral_bandwidth(S=S, sr=ASR).mean()),
        "spectral_rolloff_85_hz": float(librosa.feature.spectral_rolloff(S=S, sr=ASR, roll_percent=0.85).mean()),
        "spectral_flatness": float(librosa.feature.spectral_flatness(S=S).mean()),
        "spectral_contrast_db": [round(float(v), 2) for v in librosa.feature.spectral_contrast(S=S, sr=ASR).mean(1)],
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y, hop_length=512).mean()),
        "mfcc_mean": [round(float(v), 2) for v in mfcc.mean(1)], "mfcc_std": [round(float(v), 2) for v in mfcc.std(1)],
        "percussive_ratio": float(P.sum() / max(H.sum() + P.sum(), 1e-9)),
        "harmonic_ratio": float(H.sum() / max(H.sum() + P.sum(), 1e-9)),
        "onset_strength_mean": float(env.mean()),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}, S, logmel


def bar_features(S: np.ndarray, logmel: np.ndarray, y: np.ndarray, grid: dict, fd: float, dur: float) -> dict:
    bar = 4 * grid["period"]
    n = max(int((dur - fd) // bar), 0)
    fps = ASR / 512
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    lo, mi = freqs < 150, (freqs >= 150) & (freqs < 2500)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=ASR)
    rows = []
    for k in range(n):
        a, b = int((fd + k * bar) * fps), max(int((fd + (k + 1) * bar) * fps), int((fd + k * bar) * fps) + 1)
        s = S[:, a:b] ** 2
        rows.append([_db(s.sum(0).mean()), _db(s[lo].sum(0).mean()), _db(s[mi].sum(0).mean()),
                     _db(s[~lo & ~mi].sum(0).mean()), *logmel[:, a:b].mean(1), *chroma[:, a:b].mean(1)])
    X = np.array(rows) if rows else np.zeros((0, 4 + 64 + 12))
    return {"n_bars": n, "X": X, "bar_sec": bar}


def _norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def _label_margin(s: dict, te: float, tb: float) -> float:
    """Distance (0..1 scale) of a section's features from the thresholds that decided its label."""
    e, b = s["energy"] - te, s["bass"] - tb
    return {"drop": min(e, b), "breakdown": -b, "build": s["energy_trend"] - 0.15,
            "groove": min(b, 0.15 - s["energy_trend"]), "intro": max(-e, -b), "outro": max(-e, -b)}[s["label"]]


def structure(bf: dict, fd: float, grid_conf: float | None = None) -> dict:
    X, n, bar = bf["X"], bf["n_bars"], bf["bar_sec"]
    if n < 8:
        return {"sections": [], "phrase_bars": None, "energy_curve": [], "bass_curve": [], "onset_curve": [], "loop_candidates": [],
                "confidence": 0.0, "confidence_components": {"bars_available": round(n / 8, 3)}}
    energy, bass = _norm01(X[:, 0]), _norm01(X[:, 1])
    F = X[:, 4:]
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-6)
    k = int(np.clip(round(n / 14), 3, 12))
    bounds = [0, *sorted(int(b) for b in librosa.segment.agglomerative(F.T, k) if 0 < b < n), n]
    raw_bounds = {int(b): int(b) for b in bounds}
    snapped = []
    for b in bounds[1:-1]:  # a fill bar before a change clusters with the next section: pull the cut onto the 4-bar grid
        snapped.append(b + (1 if b % 4 == 3 else -1 if b % 4 == 1 else 0))
        raw_bounds[snapped[-1]] = int(b)
    bounds = [bounds[0], *sorted(set(snapped)), n]
    merged = [bounds[0]]  # drop sections shorter than 4 bars
    for b in bounds[1:-1]:
        if b - merged[-1] >= 4:
            merged.append(b)
    if n - merged[-1] < 4 and len(merged) > 1:
        merged.pop()
    merged.append(n)
    secs = []
    for a, b in zip(merged, merged[1:]):
        secs.append({"start_bar": a, "end_bar": b, "bars": b - a, "raw_start_bar": raw_bounds.get(a, a), "start_sec": round(fd + a * bar, 2),
                     "end_sec": round(fd + b * bar, 2), "energy": round(float(energy[a:b].mean()), 3),
                     "bass": round(float(bass[a:b].mean()), 3),
                     "energy_trend": round(float(np.polyfit(np.arange(b - a), energy[a:b], 1)[0] * (b - a)) if b - a > 1 else 0, 3)})
    med_e, med_b = np.median([s["energy"] for s in secs]), np.median([s["bass"] for s in secs])
    for i, s in enumerate(secs):
        high, bassy = s["energy"] >= max(med_e, 0.45), s["bass"] >= max(med_b * 0.8, 0.35)
        if i == 0 and not (high and bassy):
            s["label"] = "intro"
        elif i == len(secs) - 1 and not (high and bassy):
            s["label"] = "outro"
        elif high and bassy:
            s["label"] = "drop"
        elif not bassy:
            s["label"] = "breakdown"
        elif s["energy_trend"] > 0.15:
            s["label"] = "build"
        else:
            s["label"] = "groove"
    for i in range(len(secs) - 1):  # a rising section that leads into a drop is a build
        if secs[i]["label"] in ("groove", "breakdown") and secs[i]["energy_trend"] > 0.2 and secs[i + 1]["label"] == "drop":
            secs[i]["label"] = "build"
    for i, s in enumerate(secs):
        a, b = s["start_bar"], s["end_bar"]
        mu = F[a:b].mean(0)
        contrast = None
        if i:
            pa, pb = secs[i - 1]["start_bar"], secs[i - 1]["end_bar"]
            within = np.sqrt(np.mean(np.r_[((F[pa:pb] - F[pa:pb].mean(0)) ** 2).sum(1), ((F[a:b] - mu) ** 2).sum(1)]))
            between = float(np.linalg.norm(mu - F[pa:pb].mean(0)))
            contrast = between / max(between + within, 1e-9)
        t_e, t_b = max(med_e, 0.45), max(med_b * 0.8, 0.35)
        attach(s, {"boundary_contrast": contrast, "label_margin": unit(_label_margin(s, t_e, t_b), 0, 0.25)})
    lens = [s["bars"] for s in secs[:-1] if s["bars"] >= 4] or [16]
    phrase = int(min([4, 8, 16, 32, 64], key=lambda c: abs(np.log2(c) - np.log2(np.median(lens)))))
    phrase_fit = float(np.mean([s["bars"] % phrase == 0 for s in secs[:-1]])) if len(secs) > 1 else 1.0
    # loop candidates: a window of m bars that repeats in the next m bars
    Fm = X[:, 4:68] - X[:, 4:68].mean(0)
    cands = []
    for m in (1, 2, 4, 8):
        sims = []
        for k0 in range(0, n - 2 * m + 1):
            a, b = Fm[k0 : k0 + m].ravel(), Fm[k0 + m : k0 + 2 * m].ravel()
            sims.append(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)))
        taken = []
        for k0 in np.argsort(sims)[::-1]:
            if sims[k0] < 0.9 or len(taken) >= 2:
                break
            if all(abs(k0 - t) >= m for t in taken) and energy[k0 : k0 + m].mean() > 0.2:
                taken.append(int(k0))
                cands.append(attach({"start_bar": int(k0), "bars": m, "start_sec": round(fd + k0 * bar, 2),
                                     "repeat_similarity": round(sims[k0], 3), "energy": round(float(energy[k0 : k0 + m].mean()), 3)},
                                    {"repeat_similarity": unit(sims[k0], 0.9, 1.0), "energy": float(energy[k0 : k0 + m].mean()),
                                     "grid_confidence": grid_conf}, "product"))
    return attach({"sections": secs, "phrase_bars": phrase, "energy_curve": [round(float(v), 3) for v in energy],
                   "bass_curve": [round(float(v), 3) for v in bass], "onset_curve": [],
                   "loop_candidates": sorted(cands, key=lambda c: -c["repeat_similarity"])[:8]},
                  {"section_confidence": float(np.mean([s["confidence"] for s in secs])), "phrase_fit": phrase_fit})


def cue_points(st: dict, grid: dict, fd: float, dur: float, n_bars: int, grid_conf: float | None = None,
               downbeat_conf: float | None = None) -> dict:
    bar = 4 * grid["period"]
    secs = st["sections"]
    cues = [attach({"name": "first_downbeat", "bar": 0, "sec": round(fd, 3)}, {"downbeat": downbeat_conf, "grid": grid_conf})]
    for s in secs:
        if s["label"] in ("drop", "breakdown", "build", "outro") and not any(c["name"].startswith(s["label"]) for c in cues):
            cues.append(attach({"name": s["label"], "bar": s["start_bar"], "sec": s["start_sec"]},
                               {"section": s["confidence"], "grid": grid_conf, "downbeat": downbeat_conf}))
    phrase = st["phrase_bars"] or 16
    phrase_fit = st.get("confidence_components", {}).get("phrase_fit")
    intro = secs[0] if secs and secs[0]["label"] == "intro" else None
    mix_in = attach({"bar": 0, "sec": round(fd, 3), "why": "start of track; intro is the natural blend zone"},
                    {"downbeat": downbeat_conf, "grid": grid_conf, "section": intro["confidence"] if intro else None})
    outro = next((s for s in reversed(secs) if s["label"] == "outro"), None)
    out_bar = outro["start_bar"] if outro else max(n_bars - phrase, 0)
    out_bar = max(n_bars - (-(-(n_bars - out_bar) // 8)) * 8, 0) if n_bars - out_bar > 0 else out_bar
    mix_out = attach({"bar": int(out_bar), "sec": round(fd + out_bar * bar, 2),
                      "why": "start of outro / last phrase: start the next track here and blend over the remaining bars"},
                     {"section": outro["confidence"] if outro else 0.0, "phrase_fit": phrase_fit, "grid": grid_conf})
    return {"cue_points": cues, "mix_in": mix_in, "mix_out": mix_out,
            "intro_bars": secs[0]["bars"] if secs and secs[0]["label"] == "intro" else 0,
            "outro_bars": outro["bars"] if outro else n_bars - out_bar}


def waveform(audio: np.ndarray, points: int) -> dict:
    mono = audio.mean(0)
    low, mid, high = split_bands(mono[None])
    n = mono.shape[0]
    edges = np.linspace(0, n, points + 1).astype(int)
    pk = [float(np.abs(mono[a:b]).max()) if b > a else 0.0 for a, b in zip(edges, edges[1:])]
    rm = lambda x: [float(np.sqrt((x[0, a:b] ** 2).mean())) if b > a else 0.0 for a, b in zip(edges, edges[1:])]
    return {"points": points, "peak": np.round(pk, 3).tolist(), "low": np.round(rm(low), 3).tolist(),
            "mid": np.round(rm(mid), 3).tolist(), "high": np.round(rm(high), 3).tolist()}


def analyze_audio(audio: np.ndarray, path: str = "", bitrate_kbps: int | None = None) -> dict:
    dur = audio.shape[1] / SR
    y = soxr.resample(audio.mean(0), SR, ASR).astype(np.float32)
    g = beat_grid(y)
    fd, db_comps = downbeat_estimate(y, g["t0"], g["period"])
    env = g["env"]
    bpm = 60.0 / g["period"]
    at_grid = np.interp((g["t0"] + np.arange(int((dur - g["t0"]) / g["period"])) * g["period"]) * ASR / HOP,
                        np.arange(len(env)), env)
    grid_strength = float(at_grid.mean() / max(env.mean(), 1e-9))
    tempo = {
        "bpm": round(bpm, 3), "bpm_rounded": round(bpm), "first_downbeat_sec": round(fd, 4), "beats_per_bar": 4,
        "beat_period_sec": round(g["period"], 6), "grid_stable": bool(g["grid_stable"]),
        "grid_strength": round(grid_strength, 3), "phase_drift_ms": [round(o * 1000) for o in g["offsets"]],
        "alternatives": {"half_time": round(bpm / 2, 2), "double_time": round(bpm * 2, 2)},
        "n_beats": int(len(at_grid)), "n_bars": int((dur - fd) // (4 * g["period"])),
    }
    offs = np.abs(g["offsets"])
    attach(tempo, {"grid_strength": unit(grid_strength, 1.0, 3.0), "phase_agreement": float((offs < 0.03).mean()) if len(offs) else 0.0,
                   "phase_evidence": len(offs) / 8})
    tempo["downbeat_confidence"], tempo["downbeat_confidence_components"] = combine({"phase_contrast": unit(db_comps["phase_contrast"], 0, 0.25)})
    key = key_analysis(y, dur)
    key["harmonic_neighbours"] = harmonic_neighbours(key["camelot"])
    loud = loudness_block(audio, path, dur)
    spec = spectrum_block(audio.mean(0), bitrate_kbps)
    ff, S, logmel = frame_features(y, env)
    bf = bar_features(S, logmel, y, g, fd, dur)
    st = structure(bf, fd, tempo["confidence"])
    cues = cue_points(st, g, fd, dur, bf["n_bars"], tempo["confidence"], tempo["downbeat_confidence"])
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=ASR, hop_length=HOP)
    onset_density = float(onsets.size / dur)
    if st["sections"]:
        on_sec = librosa.frames_to_time(onsets, sr=ASR, hop_length=HOP)
        st["onset_curve"] = [round(float(((on_sec >= fd + k * bf["bar_sec"]) & (on_sec < fd + (k + 1) * bf["bar_sec"])).sum() / bf["bar_sec"]), 2)
                             for k in range(bf["n_bars"])]
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
                         {"component_agreement": 1.0 - unit(float(np.std(terms)), 0, 0.5), "loudness_finite": float(np.isfinite(loud["integrated_lufs"])),
                          "grid_clarity": clar}),
        "structure": st, "mix_points": cues, "duration_sec": round(dur, 3),
        "_waveform": waveform(audio, int(min(1200, max(dur * 2, 50)))),
    }
