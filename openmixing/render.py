"""Offline set renderer: clips -> per-deck audio (tempo, loops, EQ, filter, fader, fx) -> summed, limited, written."""
from __future__ import annotations

import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .decode import decode
from .dsp import eq as eqm
from .dsp import fx as fxm
from .dsp import limiter
from .dsp import stretch
from .models import SR
from .setmodel import STEM_LANES, Clip, Lane, SetModel, TempoMap, resolve_starts

TAIL = int(fxm.MAX_TAIL_S * SR)
FADE = int(0.001 * SR)


class AudioCache:
    def __init__(self, size: int = 6):
        self.size, self._d = size, OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._d:
            self._d.move_to_end(path)
            return self._d[path]
        a = decode(path)
        self._d[path] = a
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return a


def segments(clip: Clip) -> list[tuple]:
    """Output-beat segments: ("play", o0, o1, src_beat) and ("loop", o0, o1, src_beat, size)."""
    clip.validate_loops()
    segs, cur_o, cur_s = [], 0.0, clip.src_start_beat
    for L in sorted(clip.loops, key=lambda l: l.beat):
        if L.beat >= clip.length_beats:
            break
        if L.beat > cur_o:
            segs.append(("play", cur_o, L.beat, cur_s))
            cur_s += L.beat - cur_o
            cur_o = L.beat
        end = min(L.beat + L.length_beats, clip.length_beats)
        segs.append(("loop", cur_o, end, cur_s, L.size_beats))
        run = end - cur_o
        r = run % L.size_beats
        if L.size_beats - r < 1e-9:
            r = 0.0
        cur_s += run if L.roll else r
        cur_o = end
    if cur_o < clip.length_beats:
        segs.append(("play", cur_o, clip.length_beats, cur_s))
    return segs


def expand_window(clip: Clip, wa: float, wb: float) -> tuple[float, float]:
    """Pull the window start back so fx / loops already running at wa are rendered from their start."""
    wa = max(wa - 2.0, 0.0)
    changed = True
    while changed:
        changed = False
        for ev in [*clip.fx, *clip.loops]:
            dur = getattr(ev, "duration_beats", None) or ev.length_beats
            if ev.beat < wa < ev.beat + dur + 1e-9:
                wa, changed = ev.beat, True
    return wa, min(wb, clip.length_beats)


def stem_levels(clip: Clip) -> dict[str, Lane]:
    """The stem faders a stems-clip actually needs: a stem that is inaudible over the whole clip is not loaded or stretched."""
    x = np.linspace(min(0.1, clip.length_beats), clip.length_beats, max(int(clip.length_beats * 4), 2))  # past the 0.02-beat de-click ramp
    return {n: clip.lanes[l] for n, l in STEM_LANES.items() if clip.lanes[l].sample(x).max() > 1e-3}


def load_audio(clip: Clip, info: dict, cache: "AudioCache"):
    """Full mix as an array, or {stem: array} for a stems clip."""
    if not clip.stems:
        return cache.get(info["path"])
    if not info.get("stems"):
        raise ValueError(f"clip {clip.id} plays stems but {info['file']} has " + (
            "only the old 3-stem layout: call separate_stems(force=True) to re-split into vocals / drums / bass / other"
            if info.get("stems_legacy") else "none: call separate_stems first"))
    return {n: cache.get(info["stems"][n]) for n in stem_levels(clip)}


def render_clip(clip: Clip, info: dict, audio, tm: TempoMap, win: tuple[int, int] | None = None) -> np.ndarray:
    """Deck audio for sample window `win` of the clip timeline, plus TAIL samples of fx ring-out.

    `audio` is the decoded track, or {stem: decoded stem} for a clip with `stems` on: each stem is played through the same
    tempo / loop engine, scaled by its stem fader, and summed before the EQ / filter / fader / fx."""
    N = tm.beat_to_sample(clip.length_beats)
    na, nb = win or (0, N)
    na, nb = max(na, 0), min(nb, N)
    W = nb - na
    buf = np.zeros((2, W + TAIL), np.float32)
    if W <= 0:
        return buf
    gain = np.float32(10 ** ((float(info.get("gain_db", 0.0)) + clip.gain_db) / 20))
    ob = tm.sec_to_beat(np.arange(na, nb) / SR)
    main = buf[:, :W]
    stem_tails = np.zeros((2, TAIL), np.float32)
    if isinstance(audio, dict):
        for name, a in audio.items():
            lane = clip.lanes[STEM_LANES[name]]
            sig = play_source(clip, info, a, tm, na, nb) * (1.0 if lane.is_neutral() else lane.sample(ob))
            evs = [e for e in clip.fx if e.stem == name]
            if evs:  # effects aimed at this stem only: applied before it is summed, so the other stems stay dry
                sb = np.zeros((2, W + TAIL), np.float32)
                sb[:, :W] = sig
                apply_fx(sb, evs, clip, tm, na, W)
                sig, stem_tails = sb[:, :W], stem_tails + sb[:, W:]
            main += sig
    else:
        if any(e.stem for e in clip.fx):
            raise ValueError(f"clip {clip.id} has stem fx but plays the full mix: set stems=True (needs separate_stems)")
        main += play_source(clip, info, audio, tm, na, nb)
    main *= gain
    buf[:, W:] += stem_tails * gain
    lanes = clip.lanes

    def env(name):
        return None if lanes[name].is_neutral() else lanes[name].sample(ob)

    main[:] = eqm.apply_eq(main, env("eq_low"), env("eq_mid"), env("eq_high"))
    if (f := env("filter")) is not None:
        main[:] = eqm.apply_filter_knob(main, f)
    if (v := env("volume")) is not None:
        main *= v
    apply_fx(buf, [e for e in clip.fx if not e.stem], clip, tm, na, W)
    return buf


def apply_fx(buf: np.ndarray, events, clip: Clip, tm: TempoMap, na: int, W: int) -> None:
    """Run fx events over a deck (or single-stem) buffer that has TAIL samples of room after the W window samples."""
    for ev in sorted(events, key=lambda e: e.beat):
        s0 = tm.beat_to_sample(ev.beat) - na
        s1 = tm.beat_to_sample(min(ev.beat + ev.duration_beats, clip.length_beats)) - na
        if s1 <= 0 or s0 >= W:
            continue
        beat_sec = 60.0 / float(tm.bpm_at(ev.beat))
        fxm.apply(buf, ev.fx, s0, min(s1, W), beat_sec, ev.beats, ev.wet, ev.params)


def play_source(clip: Clip, info: dict, audio: np.ndarray, tm: TempoMap, na: int, nb: int) -> np.ndarray:
    """One source (full mix or a stem) through the clip's tempo map and loops for output samples [na, nb): (2, nb - na)."""
    src_bpm, fd = float(info["bpm"]), float(info.get("first_downbeat", 0.0))
    buf = np.zeros((2, nb - na), np.float32)
    kl, semi = clip.key_lock, clip.semitones
    chunk = max(int(2 * 60.0 / src_bpm * SR), 4096)

    def src_sample(beat: float) -> float:
        return (fd + beat * 60.0 / src_bpm) * SR

    prev_kind = None
    for seg in segments(clip):
        kind, o0, o1, s0 = seg[:4]
        n0, n1 = tm.beat_to_sample(o0), tm.beat_to_sample(o1)
        a, b = max(n0, na), min(n1, nb)
        if b <= a:
            prev_kind = kind
            continue
        if kind == "play":
            o_a = float(tm.sec_to_beat(a / SR))
            i0, i1 = int(o0 * tm.RES), min(int(np.ceil(o1 * tm.RES)) + 1, len(tm.bpm))
            if np.ptp(tm.bpm[i0:i1]) < 1e-6:
                speed = float(tm.bpm_at(o_a)) / src_bpm
            else:
                speed = tm.bpm_at(tm.sec_to_beat(np.arange(a, b) / SR)) / src_bpm
            y = stretch.play(audio, src_sample(s0 + (o_a - o0)), speed, b - a, semi, kl, chunk)
            if prev_kind == "loop":
                f = min(FADE, y.shape[1] // 4)
                if f:
                    y[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
            buf[:, a - na : b - na] += y
        else:
            size = seg[4]
            speed = float(tm.bpm_at(o0)) / src_bpm
            body_n = tm.beat_to_sample(o0 + size) - n0 + 2
            body = stretch.play(audio, src_sample(s0), speed, body_n, semi, kl, chunk)
            j = 0
            while o0 + j * size < o1 - 1e-9:
                oj = o0 + j * size
                nj, nj1 = tm.beat_to_sample(oj), tm.beat_to_sample(min(oj + size, o1))
                j += 1
                lo, hi = max(nj, na), min(nj1, nb)
                if hi <= lo:
                    continue
                piece = body[:, : nj1 - nj].copy()
                f = min(FADE, piece.shape[1] // 4)
                if f:
                    piece[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
                    piece[:, -f:] *= np.linspace(1, 0, f, dtype=np.float32)
                buf[:, lo - na : hi - na] += piece[:, lo - nj : hi - nj]
        prev_kind = kind
    return buf


def _write(path: Path, fmt: str):
    if fmt == "flac":
        return sf.SoundFile(path, "w", SR, 2, format="FLAC", subtype="PCM_24")
    return sf.SoundFile(path, "w", SR, 2, format="WAV", subtype="PCM_24")


def _low_curve(seg: np.ndarray, offset: int, n_blocks: int, blk: int, prev: np.ndarray | None) -> np.ndarray:
    """Low-band level of one deck's contribution per block, placed on the render's block grid (-120 dB where the deck is silent)."""
    out = np.full(n_blocks, -120.0, np.float32) if prev is None else prev
    low = eqm.split_bands(seg)[0]
    for k in range(0, seg.shape[1], blk):
        j = (offset + k) // blk
        if j >= n_blocks:
            break
        part = low[:, k : k + blk]
        out[j] = max(out[j], 10 * np.log10(float((part ** 2).mean()) + 1e-12))
    return out


def measure(path: Path) -> dict:
    """Integrated loudness / LRA / true peak via ffmpeg's ebur128."""
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    t = r.stderr[r.stderr.rfind("Summary:"):]
    num = lambda pat: (float(m.group(1)) if (m := re.search(pat, t)) else None)
    return {"integrated_lufs": num(r"I:\s+(-?[\d.]+) LUFS"), "loudness_range_lu": num(r"LRA:\s+(-?[\d.]+) LU"),
            "true_peak_dbtp": num(r"Peak:\s+(-?[\d.]+) dBFS")}


def timeline(model: SetModel, info: Callable[[str], dict]) -> list[dict]:
    """Resolved clip spans on the set timeline (seconds)."""
    starts = resolve_starts(model, lambda t: float(info(t)["bpm"]))
    rows = []
    for c in model.clips:
        tm = TempoMap(c, float(info(c.track)["bpm"]))
        dur = float(tm.beat_to_sec(c.length_beats))
        rows.append({"clip": c.id, "track": c.track, "start_sec": round(starts[c.id], 3),
                     "end_sec": round(starts[c.id] + dur, 3), "duration_sec": round(dur, 3)})
    return rows


def render_set(model: SetModel, info: Callable[[str], dict], out_path: Path, start_sec: float | None = None,
               end_sec: float | None = None, fmt: str = "flac", master_gain_db: float = 0.0, ceiling_db: float = -1.5,
               progress: Callable[[int, int, str], None] | None = None) -> dict:
    if not model.clips:
        raise ValueError("the set has no clips")
    rows = {r["clip"]: r for r in timeline(model, info)}
    t0 = min(r["start_sec"] for r in rows.values()) if start_sec is None else start_sec
    t1 = max(r["end_sec"] for r in rows.values()) if end_sec is None else end_sec
    if t1 <= t0:
        raise ValueError(f"empty range {t0}..{t1}")
    s_lo, s_hi = int(round(t0 * SR)), int(round(t1 * SR))
    cache = AudioCache()
    work = []
    for c in model.clips:
        r = rows[c.id]
        if r["end_sec"] + fxm.MAX_TAIL_S < t0 or r["start_sec"] > t1:
            continue
        tm = TempoMap(c, float(info(c.track)["bpm"]))
        cs = int(round(r["start_sec"] * SR))
        wa = max(float(tm.sec_to_beat(max(t0 - r["start_sec"], 0.0))), 0.0)
        wb = float(tm.sec_to_beat(min(t1 - r["start_sec"], r["duration_sec"])))
        if start_sec is not None or end_sec is not None:
            wa, wb = expand_window(c, wa, wb)
        na, nb = tm.beat_to_sample(wa), tm.beat_to_sample(wb)
        if nb > na:
            work.append((cs + na, c, tm, cs, na, nb))
    work.sort(key=lambda w: w[0])
    low_blk = SR // 4
    low_db: dict[str, np.ndarray] = {}   # per clip: low band (<~200 Hz) level in dBFS per 0.25 s block of the render

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mg = np.float32(10 ** (master_gain_db / 20))
    acc, acc_at = np.zeros((2, 0), np.float32), s_lo
    raw = out_path.with_suffix(".raw.wav")
    with sf.SoundFile(raw, "w", SR, 2, format="WAV", subtype="FLOAT") as f:
        def flush(upto: int):
            nonlocal acc, acc_at
            n = min(max(upto - acc_at, 0), acc.shape[1])
            if n <= 0:
                return
            f.write((acc[:, :n] * mg).T)
            acc, acc_at = acc[:, n:], acc_at + n

        for i, (g0, c, tm, cs, na, nb) in enumerate(work):
            if progress:
                progress(i, len(work), f"rendering {c.id}")
            flush(g0)
            deck = render_clip(c, info(c.track), load_audio(c, info(c.track), cache), tm, (na, nb))
            lo = max(g0, s_lo)
            need = (g0 + deck.shape[1]) - acc_at
            if need > acc.shape[1]:
                acc = np.pad(acc, ((0, 0), (0, need - acc.shape[1])))
            seg = deck[:, lo - g0 :][:, : max(s_hi - lo, 0)]
            acc[:, lo - acc_at : lo - acc_at + seg.shape[1]] += seg
            if seg.shape[1]:
                low_db[c.id] = _low_curve(seg, lo - s_lo, (s_hi - s_lo + low_blk - 1) // low_blk, low_blk, low_db.get(c.id))
        flush(s_hi)
        total = s_hi - s_lo
        if acc_at < s_hi:  # silence up to the requested end
            f.write(np.zeros((s_hi - acc_at, 2), np.float32))

    # second pass: master limiter in 60 s blocks with 1 s of context on each side
    peaks: list[float] = []
    blk = SR // 4
    tmp = out_path.with_suffix(".part.wav" if fmt == "mp3" else out_path.suffix + ".part")
    step, ctx = 60 * SR, SR
    with sf.SoundFile(raw) as rf, _write(tmp, "wav" if fmt == "mp3" else fmt) as wf:
        for p0 in range(0, total, step):
            c0, c1 = max(p0 - ctx, 0), min(p0 + step + ctx, total)
            rf.seek(c0)
            x = rf.read(c1 - c0, dtype="float32").T
            y = limiter.limit(x, ceiling_db)[:, p0 - c0 : p0 - c0 + min(step, total - p0)]
            wf.write(y.T)
            for i in range(0, y.shape[1], blk):
                peaks.append(round(float(np.abs(y[:, i : i + blk]).max()), 4))
    raw.unlink()
    if fmt == "mp3":
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-b:a", "320k", str(out_path)], check=True)
        tmp.unlink()
    else:
        tmp.replace(out_path)
    meta = {"path": str(out_path), "format": fmt, "start_sec": t0, "end_sec": t1, "duration_sec": round(t1 - t0, 3),
            "clips": list(rows.values()), "peaks": peaks, "peak_block_sec": blk / SR, "low_db": low_db, "low_block_sec": low_blk / SR}
    meta.update(measure(out_path))
    return meta
