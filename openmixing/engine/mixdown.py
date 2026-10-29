from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from ..core.cancel import JobCancelled
from ..core.measure import measure
from ..core.units import SR
from ..dsp import fx as fxm
from ..dsp import limiter
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap, resolve_starts
from .cache import AudioCache
from .deck import load_audio, render_clip
from .keys import clip_ducks
from .sampler import load_samples
from .segments import expand_window
from .taps import BLOCK, Meters, low_curve

MP3_KBPS = 320


def _write(path: Path, fmt: str):
    if fmt == "flac":
        return sf.SoundFile(path, "w", SR, 2, format="FLAC", subtype="PCM_24")
    return sf.SoundFile(path, "w", SR, 2, format="WAV", subtype="PCM_24")


def timeline(model: SetModel, info: Callable[[str], dict]) -> list[dict]:
    starts = resolve_starts(model, lambda t: float(info(t)["bpm"]))
    rows = []
    for c in model.clips:
        tm = TempoMap(c, float(info(c.track)["bpm"]))
        dur = float(tm.beat_to_sec(c.length_beats))
        rows.append({"clip": c.id, "track": c.track, "start_sec": round(starts[c.id], 3),
                     "end_sec": round(starts[c.id] + dur, 3), "duration_sec": round(dur, 3)})
    return rows


def _render(model: SetModel, info: Callable[[str], dict], out_path: Path, start_sec: float | None, end_sec: float | None, fmt: str,
            master_gain_db: float, ceiling_db: float, progress: Callable[[int, int, str], None] | None, cancel: Callable[[], bool],
            tmps: tuple[Path, Path, Path]) -> dict:
    if not model.clips:
        raise ValueError("the set has no clips")
    rows = {r["clip"]: r for r in timeline(model, info)}
    t0 = min(r["start_sec"] for r in rows.values()) if start_sec is None else start_sec
    t1 = max(r["end_sec"] for r in rows.values()) if end_sec is None else end_sec
    if t1 <= t0:
        raise ValueError(f"empty range {t0}..{t1}")
    s_lo, s_hi = int(round(t0 * SR)), int(round(t1 * SR))
    cache = AudioCache()
    samples = load_samples(model)
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
    low_db: dict[str, np.ndarray] = {}
    meters = Meters((s_hi - s_lo + BLOCK - 1) // BLOCK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mg = np.float32(10 ** (master_gain_db / 20))
    acc, acc_at = np.zeros((2, 0), np.float32), s_lo
    raw, tmp, enc = tmps
    with sf.SoundFile(raw, "w", SR, 2, format="WAV", subtype="FLOAT") as f:
        def flush(upto: int):
            nonlocal acc, acc_at
            n = min(max(upto - acc_at, 0), acc.shape[1])
            if n <= 0:
                return
            f.write((acc[:, :n] * mg).T)
            acc, acc_at = acc[:, n:], acc_at + n

        for i, (g0, c, tm, cs, na, nb) in enumerate(work):
            if cancel():
                raise JobCancelled()
            if progress:
                progress(i, len(work), f"rendering {c.id}")
            flush(g0)
            tap: dict = {}
            deck = render_clip(c, info(c.track), load_audio(c, info(c.track), cache), tm, (na, nb),
                               clip_ducks(model, c, tm, cs, na, nb, rows, info, cache, samples), samples, tap)
            lo = max(g0, s_lo)
            need = (g0 + deck.shape[1]) - acc_at
            if need > acc.shape[1]:
                acc = np.pad(acc, ((0, 0), (0, need - acc.shape[1])))
            seg = deck[:, lo - g0 :][:, : max(s_hi - lo, 0)]
            acc[:, lo - acc_at : lo - acc_at + seg.shape[1]] += seg
            if seg.shape[1]:
                meters.add(c, info(c.track), tm, seg, tap, lo, g0, lo - s_lo, na)
                low_db[c.id] = low_curve(seg, lo - s_lo, (s_hi - s_lo + low_blk - 1) // low_blk, low_blk, low_db.get(c.id))
        flush(s_hi)
        total = s_hi - s_lo
        if acc_at < s_hi:
            f.write(np.zeros((s_hi - acc_at, 2), np.float32))

    peaks: list[float] = []
    blk = SR // 4
    step, ctx = 60 * SR, SR
    with sf.SoundFile(raw) as rf, _write(tmp, "wav" if fmt == "mp3" else fmt) as wf:
        for p0 in range(0, total, step):
            if cancel():
                raise JobCancelled()
            c0, c1 = max(p0 - ctx, 0), min(p0 + step + ctx, total)
            rf.seek(c0)
            x = rf.read(c1 - c0, dtype="float32").T
            y = limiter.limit(x, ceiling_db)[:, p0 - c0 : p0 - c0 + min(step, total - p0)]
            wf.write(y.T)
            for i in range(0, y.shape[1], blk):
                peaks.append(round(float(np.abs(y[:, i : i + blk]).max()), 4))
    raw.unlink()
    if fmt == "mp3":
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:a", "libmp3lame", "-b:a", f"{MP3_KBPS}k", "-f", "mp3", str(enc)],
                       check=True)
        enc.replace(out_path)
        tmp.unlink()
    else:
        tmp.replace(out_path)
    meta = {"path": str(out_path), "format": fmt, "start_sec": t0, "end_sec": t1, "duration_sec": round(t1 - t0, 3),
            "clips": list(rows.values()), "peaks": peaks, "peak_block_sec": blk / SR, "low_db": low_db, "low_block_sec": low_blk / SR,
            "meters": meters.export()}
    meta.update(measure(out_path))
    return meta


def render_set(model: SetModel, info: Callable[[str], dict], out_path: Path, start_sec: float | None = None,
               end_sec: float | None = None, fmt: str = "mp3", master_gain_db: float = 0.0, ceiling_db: float = -1.5,
               progress: Callable[[int, int, str], None] | None = None, cancel: Callable[[], bool] | None = None) -> dict:
    tag = uuid.uuid4().hex[:8]
    tmps = tuple(out_path.with_name(f"{out_path.stem}.{tag}.{x}") for x in ("raw.wav", "part", "enc.part"))
    try:
        return _render(model, info, out_path, start_sec, end_sec, fmt, master_gain_db, ceiling_db, progress, cancel or (lambda: False), tmps)
    finally:
        for p in tmps:
            p.unlink(missing_ok=True)
