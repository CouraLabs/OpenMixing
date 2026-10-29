from __future__ import annotations

import numpy as np

from ..analysis import stems as stemlib
from ..core.units import BAR
from ..engine.mixdown import timeline
from ..model.lane import STEM_LANES
from ..model.rate import net_shift
from ..model.setmodel import SetModel
from ..model.tempo import TempoMap
from .compat import camelot_relation


def audit(model: SetModel, lib) -> dict:
    info = lib.effective
    rows = {r["clip"]: r for r in timeline(model, info)}
    issues: list[dict] = []
    add = lambda sev, clip, msg: issues.append({"severity": sev, "clip": clip, "message": msg})
    ordered = sorted(model.clips, key=lambda c: rows[c.id]["start_sec"])
    for c in ordered:
        e, d = info(c.track), lib.analyze(c.track)
        n_beats = d["tempo"]["n_beats"]
        end_beat = c.src_start_beat + c.length_beats + net_shift(c)
        if end_beat > n_beats + 2:
            add("error", c.id, f"plays to track beat {end_beat:g} but {e['file']} only has ~{n_beats} beats: the rest would be silence")
        if (c.src_start_beat % BAR) > 1e-6:
            add("warn", c.id, f"src_start_beat {c.src_start_beat:g} is not on a bar line: phrase alignment will be off")
        if c.stems and not e.get("stems"):
            add("error", c.id, f"plays stems but {e['file']} has none: call separate_stems (or turn stems off with edit_clip)")
        if not d["tempo"]["grid_stable"]:
            add("warn", c.id, "beat grid is not stable (live drums / tempo drift): long blends will drift; keep the overlap short")
        if d["tempo"]["grid_strength"] < 1.5:
            add("warn", c.id, "weak beat grid confidence: verify bpm / first_downbeat with override_track")
        tm = TempoMap(c, e["bpm"])
        pct = (tm.bpm / e["bpm"] - 1) * 100
        if np.abs(pct).max() > 8:
            add("warn", c.id, f"tempo stretched up to {np.abs(pct).max():.1f}% from native: expect audible artifacts")
        for lp in c.loops:
            if lp.size_beats > lp.length_beats + 1e-9:
                add("warn", c.id, f"loop {lp.id} is longer than the time it is engaged")
        for r in c.rate:
            if any(abs(r.beat - lp.beat - lp.length_beats) < 0.5 or abs(lp.beat - r.beat - r.duration_beats) < 0.5 for lp in c.loops):
                add("warn", c.id, f"rate event {r.id} is within half a beat of a loop: the two fades will stack")
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            ra, rb = rows[a.id], rows[b.id]
            lo, hi = max(ra["start_sec"], rb["start_sec"]), min(ra["end_sec"], rb["end_sec"])
            if hi - lo < 0.5:
                if b is ordered[i + 1] and rb["start_sec"] - ra["end_sec"] > 0.5:
                    add("warn", b.id, f"gap of {rb['start_sec'] - ra['end_sec']:.1f}s of silence between {a.id} and {b.id}")
                continue
            ea, eb = info(a.track), info(b.track)
            dist, rel = camelot_relation(ea["camelot"], eb["camelot"])
            semis = a.semitones != 0 or b.semitones != 0
            if dist >= 3 and not semis:
                add("warn", b.id,
                    f"overlaps {a.id} for {hi - lo:.0f}s but keys clash ({ea['camelot']} vs {eb['camelot']}): "
                    "shorten the overlap, filter out harmonic content, or pick a closer key")
            ta, tb = TempoMap(a, ea["bpm"]), TempoMap(b, eb["bpm"])
            ts = np.arange(lo, hi, 0.25)
            ba, bb = ta.sec_to_beat(ts - ra["start_sec"]), tb.sec_to_beat(ts - rb["start_sec"])
            ga = a.lanes["eq_low"].sample(ba) > -20
            gb = b.lanes["eq_low"].sample(bb) > -20
            if a.stems:
                ga &= a.lanes[STEM_LANES[stemlib.low_stem(ea)]].sample(ba) > 0.3
            if b.stems:
                gb &= b.lanes[STEM_LANES[stemlib.low_stem(eb)]].sample(bb) > 0.3
            va = a.lanes["volume"].sample(ba) > 0.3
            vb = b.lanes["volume"].sample(bb) > 0.3
            clash = ga & gb & va & vb
            if clash.mean() > 0.15:
                first, last = ts[clash][0], ts[clash][-1]
                add("warn", b.id,
                    f"bass clash with {a.id}: both low bands open and audible for {clash.sum() * 0.25:.0f}s (set time {first:.0f}-{last:.0f}s): "
                    "kill one low band (apply_transition bass_swap does this)")
            if a.stems and b.stems:
                vox = ((a.lanes["stem_vocals"].sample(ba) > 0.3) & (b.lanes["stem_vocals"].sample(bb) > 0.3) & va & vb)
                if vox.mean() > 0.15:
                    add("warn", b.id, f"vocal clash with {a.id}: both vocal stems audible for {vox.sum() * 0.25:.0f}s: fade one with set_stem")
            if hi - lo > 90:
                add("info", b.id, f"long overlap with {a.id} ({hi - lo:.0f}s)")
            st = model.clip(b.id).start
            if "anchor" in st and st["anchor"] == a.id and st["anchor_beat"] % 16 > 1e-6:
                add("warn", b.id, f"enters {a.id} at bar {st['anchor_beat'] / BAR:g}: not on a 4-bar phrase boundary")
    sev = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda i: sev[i["severity"]])
    return {"ok": not any(i["severity"] == "error" for i in issues), "n_issues": len(issues), "issues": issues,
            "timeline": list(rows.values())}
