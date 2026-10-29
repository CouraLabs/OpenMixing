from __future__ import annotations

import librosa
import numpy as np
import pyloudnorm
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

from ..core.measure import measure
from ..core.units import SR
from ..dsp.eq import split_bands
from .analyze import ASR, TARGET_LUFS
from .confidence import attach, unit

BANDS = {"sub": (20, 60), "bass": (60, 250), "low_mid": (250, 500), "mid": (500, 2000), "high_mid": (2000, 6000),
         "high": (6000, 20000)}


def _db(x, floor=1e-10):
    return 10 * np.log10(np.maximum(x, floor))


def loudness_block(audio: np.ndarray, path: str, dur: float | None = None) -> dict:
    mono = audio.mean(0)
    lufs = pyloudnorm.Meter(SR).integrated_loudness(audio.T)
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((mono ** 2).mean()))
    tp = measure(path) if path else {}
    gain = float(np.clip(TARGET_LUFS - lufs, -12, 12)) if np.isfinite(lufs) else 0.0
    fr = np.sqrt(uniform_filter1d(mono ** 2, 480, mode="nearest")[240 : len(mono) - 239 : 480])
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


def frame_features(y: np.ndarray, env: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
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
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}, S, logmel, H


def bar_features(S: np.ndarray, logmel: np.ndarray, y: np.ndarray, grid: dict, fd: float, dur: float) -> dict:
    bar = 4 * grid["period"]
    n = max(int((dur - fd) // bar), 0)
    fps = ASR / 512
    freqs = librosa.fft_frequencies(sr=ASR, n_fft=2048)
    lo, mi = freqs < 150, (freqs >= 150) & (freqs < 2500)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=ASR)
    hf = S[freqs >= 4000]
    cols = np.stack([librosa.feature.spectral_centroid(S=S, sr=ASR)[0], librosa.feature.spectral_flatness(S=hf)[0],
                     np.r_[0.0, np.maximum(np.diff(hf, axis=1), 0).sum(0)], _db((S ** 2).sum(0))])
    rows = []
    for k in range(n):
        a, b = int((fd + k * bar) * fps), max(int((fd + (k + 1) * bar) * fps), int((fd + k * bar) * fps) + 1)
        s = S[:, a:b] ** 2
        rows.append([_db(s.sum(0).mean()), _db(s[lo].sum(0).mean()), _db(s[mi].sum(0).mean()),
                     _db(s[~lo & ~mi].sum(0).mean()), *logmel[:, a:b].mean(1), *chroma[:, a:b].mean(1),
                     *cols[:3, a:b].mean(1), float(np.percentile(cols[3, a:b], 90))])
    X = np.array(rows) if rows else np.zeros((0, 4 + 64 + 12 + 4))
    return {"n_bars": n, "X": X, "bar_sec": bar}


def waveform(audio: np.ndarray, points: int) -> dict:
    mono = audio.mean(0)
    low, mid, high = split_bands(mono[None])
    n = mono.shape[0]
    edges = np.linspace(0, n, points + 1).astype(int)
    pk = [float(np.abs(mono[a:b]).max()) if b > a else 0.0 for a, b in zip(edges, edges[1:])]
    rm = lambda x: [float(np.sqrt((x[0, a:b] ** 2).mean())) if b > a else 0.0 for a, b in zip(edges, edges[1:])]
    return {"points": points, "peak": np.round(pk, 3).tolist(), "low": np.round(rm(low), 3).tolist(),
            "mid": np.round(rm(mid), 3).tolist(), "high": np.round(rm(high), 3).tolist()}
