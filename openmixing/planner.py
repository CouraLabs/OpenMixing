"""DJ judgement: track compatibility, ordering, transition templates and set audit."""
from __future__ import annotations

import numpy as np

from . import stems as stemlib
from .models import SR
from .render import timeline
from .setmodel import STEM_LANES, Clip, FxEvent, LoopEvent, SetModel, TempoMap, new_id

BAR = 4.0  # beats per bar


# ------------------------------------------------------------------ compatibility
def camelot_relation(a: str, b: str) -> tuple[int, str]:
    """(distance on the Camelot wheel, description). 0 identical, 1 smooth, 2 energy shift, 3+ clash."""
    na, ma, nb, mb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    if a == b:
        return 0, "same key"
    if na == nb:
        return 1, "relative major/minor"
    step = min((na - nb) % 12, (nb - na) % 12)
    if ma == mb:
        return step, {1: "adjacent on the wheel (smooth)", 2: "two steps (energy shift)", 7: "perfect-fifth-ish energy boost"}.get(step, f"{step} steps apart")
    return step + 1, f"{step} steps and mode change"


def tempo_match(bpm_from: float, bpm_to: float) -> dict:
    """Stretch needed to play `bpm_to` at `bpm_from`, considering half/double time."""
    best = min((bpm_to * f for f in (0.5, 1.0, 2.0)), key=lambda b: abs(b / bpm_from - 1))
    pct = (bpm_from / best - 1) * 100
    return {"bpm_from": round(bpm_from, 2), "bpm_to": round(bpm_to, 2), "ratio_used": round(best / bpm_to, 3),
            "stretch_percent": round(pct, 2),
            "verdict": "seamless" if abs(pct) <= 3 else "ok (use a tempo ramp)" if abs(pct) <= 8 else "hard (large stretch; prefer a cut / echo-out)"}


def compat(ea: dict, eb: dict) -> dict:
    dist, rel = camelot_relation(ea["camelot"], eb["camelot"])
    tm = tempo_match(ea["bpm"], eb["bpm"])
    score = 100 - min(abs(tm["stretch_percent"]) * 6, 45) - {0: 0, 1: 4, 2: 12}.get(dist, 30) - abs(ea["energy"] - eb["energy"]) * 2
    return {"score": round(max(score, 0), 1), "key": {"from": ea["camelot"], "to": eb["camelot"], "distance": dist, "relation": rel},
            "tempo": tm, "energy": {"from": ea["energy"], "to": eb["energy"], "delta": eb["energy"] - ea["energy"]}}


def order_tracks(effs: list[dict], strategy: str = "harmonic", start: str | None = None) -> list[dict]:
    """Greedy path through the tracks. strategies: harmonic (best compat), energy_build (rising energy), bpm_ramp (rising bpm)."""
    pool = list(effs)
    if not pool:
        return []
    if start:
        cur = next(e for e in pool if e["id"] == start)
    elif strategy == "bpm_ramp":
        cur = min(pool, key=lambda e: e["bpm"])
    elif strategy == "energy_build":
        cur = min(pool, key=lambda e: e["energy"])
    else:
        cur = pool[0]
    path = [cur]
    pool.remove(cur)
    while pool:
        def cost(e):
            c = compat(path[-1], e)
            base = -c["score"]
            if strategy == "energy_build":
                base += 6 * max(path[-1]["energy"] - e["energy"], 0) - 2 * (e["energy"] - path[-1]["energy"])
            if strategy == "bpm_ramp":
                base += 3 * max(path[-1]["bpm"] - e["bpm"], 0)
            return base
        nxt = min(pool, key=cost)
        path.append(nxt)
        pool.remove(nxt)
    return path


# ------------------------------------------------------------------ templates
TEMPLATES = {
    "bass_swap": "Classic blend: incoming enters with bass killed, bass swaps at the midpoint, outgoing fades out. Best for same-genre, 16-32 bars.",
    "long_blend": "Slow equal-power blend with the incoming mids/highs eased in and bass swapped at 3/4. 32+ bars, harmonic matches only.",
    "filter_fade": "Outgoing is low-passed shut while the incoming opens from a high-passed state. Good across energy changes.",
    "echo_out": "Outgoing is thrown into an echo (dry cut) on the last bars and the incoming drops in full on the phrase boundary.",
    "drop_swap": "Hard swap on a phrase boundary: outgoing stops, incoming hits at full. For breakdown -> drop or genre jumps.",
    "loop_roll_out": "Outgoing rolls into shrinking loops (1, 1/2, 1/4, 1/8 beat) in its last bar, then the incoming drops.",
    "stem_blend": "Stem-based blend (both clips need separated stems): outgoing vocals leave first, the harmonic layers cross over, then the drums+bass swap, and the incoming vocals come in last. No bass or vocal ever doubles up; works across key clashes better than an EQ blend.",
}


LANE_NAMES = ("eq_low", "eq_mid", "eq_high", "filter", "volume", *STEM_LANES.values())


def _reset_from(clip: Clip, beat: float) -> None:
    """Drop automation at or after `beat` (the value reached just before it keeps holding)."""
    for l in LANE_NAMES:
        clip.lanes[l].points = [p for p in clip.lanes[l].points if p[0] < beat - 1e-9]


def apply_template(model: SetModel, info, out: Clip, inc: Clip, style: str, at_bar: float | None, bars: float,
                   tempo_after: str = "keep", trim_out: bool = True) -> dict:
    """Place `inc` over `out` and write the automation.

    Overwrites the incoming clip's EQ / filter / volume automation and the outgoing clip's from the
    transition start on, and moves/tempo-syncs the incoming clip. Returns a summary of what was done.
    """
    if style not in TEMPLATES:
        raise ValueError(f"style must be one of {list(TEMPLATES)}")
    nb = bars * BAR
    out_bars = out.length_beats / BAR
    at = float(out_bars - bars if at_bar is None else at_bar)
    if at < 0 or at * BAR >= out.length_beats:
        raise ValueError(f"at_bar {at} is outside clip {out.id} (0..{out_bars:g} bars)")
    a0 = at * BAR
    out_end = a0 + nb
    K = -80.0

    # --- placement + tempo lock
    inc.start = {"anchor": out.id, "anchor_beat": out_end if style == "loop_roll_out" else a0}
    tm = TempoMap(out, float(info(out.track)["bpm"]))
    sync_bpm = float(tm.bpm_at(a0))
    native = float(info(inc.track)["bpm"])
    ratio = tempo_match(sync_bpm, native)["ratio_used"]  # 0.5 / 1 / 2 when the pair is half/double time
    play_bpm = sync_bpm / ratio  # in the incoming track's own beat units
    inc.tempo.points = [[0.0, play_bpm]]
    if tempo_after == "native":
        inc.tempo.points += [[nb, play_bpm], [nb + 8 * BAR, native]]

    if style == "stem_blend":
        for c in (out, inc):
            if not info(c.track).get("stems"):
                raise ValueError(f"stem_blend needs stems for {info(c.track)['file']}: call separate_stems "
                                 + ("(force=True: it only has the old 3-stem layout)" if info(c.track).get("stems_legacy") else "first"))
            c.stems = True

    _reset_from(out, a0)
    for l in LANE_NAMES:
        inc.lanes[l].points = []
    notes = [f"{inc.id} enters {out.id} at bar {at:g} playing at {play_bpm:.2f} bpm (native {native:.2f})"]

    if style in ("bass_swap", "long_blend"):
        swap = nb * (0.5 if style == "bass_swap" else 0.75)
        inc.lanes["eq_low"].hold(K)
        inc.lanes["eq_low"].set(swap, 0.0, 0.5)
        out.lanes["eq_low"].set(a0 + swap, K, 0.5)
        if style == "long_blend":
            inc.lanes["eq_high"].hold(-12.0)
            inc.lanes["eq_high"].set(nb * 0.5, 0.0, nb * 0.25)
            inc.lanes["volume"].hold(0.0)
            inc.lanes["volume"].curve(0, nb, lambda p: np.sin(p * np.pi / 2))
            out.lanes["volume"].curve(a0, nb, lambda p: np.cos(p * np.pi / 2))
        else:
            inc.lanes["volume"].hold(0.0)
            inc.lanes["volume"].set(0, 1.0, nb * 0.4)
            out.lanes["volume"].set(a0 + nb * 0.6, 0.0, nb * 0.4)
        notes.append(f"incoming bass killed until beat {swap:g} of the blend, then the bass swaps; faders cross")
    elif style == "filter_fade":
        inc.lanes["filter"].hold(1.0)
        inc.lanes["filter"].set(nb * 0.25, 0.0, nb * 0.5)
        out.lanes["filter"].set(a0 + nb * 0.25, -1.0, nb * 0.6)
        inc.lanes["eq_low"].hold(K)
        inc.lanes["eq_low"].set(nb * 0.5, 0.0, 0.5)
        out.lanes["eq_low"].set(a0 + nb * 0.5, K, 0.5)
        out.lanes["volume"].set(a0 + nb * 0.85, 0.0, nb * 0.15)
        notes.append("outgoing low-passes shut, incoming high-pass opens; bass swap at the midpoint")
    elif style == "echo_out":
        fx_at = out_end - min(2 * BAR, nb)
        out.fx = [e for e in out.fx if not (a0 <= e.beat <= out_end)]
        out.fx.append(FxEvent(new_id("fx"), "echo_out", fx_at, out_end - fx_at, 1.0, 1.0, {}))
        out.lanes["volume"].set(out_end, 0.0, 0.5)
        inc.lanes["eq_low"].hold(K)
        inc.lanes["eq_low"].set(fx_at - a0, 0.0, 0.25)
        inc.lanes["volume"].hold(0.0)
        inc.lanes["volume"].set(fx_at - a0 - 0.5, 1.0, 0.5)
        notes.append(f"echo_out (1 beat) on {out.id} from bar {fx_at / BAR:g}; incoming fades in just before, bass opens with the echo")
    elif style == "stem_blend":
        ramp = nb * 0.125
        def fade(c, stem, frac, to, base=0.0):
            c.lanes[STEM_LANES[stem]].set(base + nb * frac, to, ramp)
        for stem in STEM_LANES:
            inc.lanes[STEM_LANES[stem]].hold(0.0)
        fade(out, "vocals", 0.0, 0.0, a0)                  # outgoing vocals leave first
        fade(inc, "other", 0.25, 1.0)                      # harmonic layers cross over
        fade(out, "other", 0.25, 0.0, a0)
        low_in_drums = any(info(c.track).get("bass_carrier") == "drums_lowband" for c in (out, inc))
        d_at = 0.5 if low_in_drums else 0.4                # drums that carry the 808 swap with the bass, never before it
        fade(inc, "drums", d_at, 1.0)                      # drums swap first, then the bass on the midpoint downbeat
        fade(out, "drums", d_at, 0.0, a0)
        inc.lanes[STEM_LANES["bass"]].set(nb * 0.5, 1.0, 0.25)
        out.lanes[STEM_LANES["bass"]].set(a0 + nb * 0.5, 0.0, 0.25)
        fade(inc, "vocals", 0.75, 1.0)                     # incoming vocals arrive last
        notes.append(f"vocals out -> other crossover (25%) -> drums swap ({d_at:.0%}) -> bass swap (50%) -> incoming vocals (75%); both clips now play stems")
        if low_in_drums:
            notes.append("the bass stem is empty on at least one track (808 sits in drums): the drums swap moves to the bass swap point")
    elif style == "drop_swap":
        out.lanes["volume"].set(out_end, 0.0, 0.05)
        inc.start = {"anchor": out.id, "anchor_beat": out_end}
        notes.append(f"{out.id} stops at bar {out_end / BAR:g}; {inc.id} hits at full level on that downbeat")
    elif style == "loop_roll_out":
        out.loops = [l for l in out.loops if l.beat + l.length_beats < a0 or l.beat >= out_end]
        t = out_end - BAR
        for size, ln in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.5)):
            out.loops.append(LoopEvent(new_id("lp"), t, size, ln, True))
            t += ln
        notes.append(f"loop roll 1 -> 1/2 -> 1/4 -> 1/8 beat on {out.id} in its last bar; {inc.id} drops at bar {out_end / BAR:g}")
    if trim_out:
        out.length_beats = min(out.length_beats, out_end)
    for c in (out, inc):
        for l in LANE_NAMES:
            c.lanes[l].points.sort(key=lambda p: p[0])
    return {"style": style, "incoming_enters_at_bar_of_outgoing": at, "bars": bars, "tempo_bpm": round(play_bpm, 3),
            "outgoing_clip_bars": out.length_beats / BAR, "notes": notes}


# ------------------------------------------------------------------ audit
def audit(model: SetModel, lib) -> dict:
    info = lib.render_info
    rows = {r["clip"]: r for r in timeline(model, info)}
    issues: list[dict] = []
    add = lambda sev, clip, msg: issues.append({"severity": sev, "clip": clip, "message": msg})
    ordered = sorted(model.clips, key=lambda c: rows[c.id]["start_sec"])
    for c in ordered:
        e, d = info(c.track), lib.analyze(c.track)
        n_beats = d["tempo"]["n_beats"]
        end_beat = c.src_start_beat + c.length_beats
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
                add("warn", b.id, f"overlaps {a.id} for {hi - lo:.0f}s but keys clash ({ea['camelot']} vs {eb['camelot']}): shorten the overlap, filter out harmonic content, or pick a closer key")
            # bass clash: both decks audible with low band open
            ta, tb = TempoMap(a, ea["bpm"]), TempoMap(b, eb["bpm"])
            ts = np.arange(lo, hi, 0.25)
            ba, bb = ta.sec_to_beat(ts - ra["start_sec"]), tb.sec_to_beat(ts - rb["start_sec"])
            ga = a.lanes["eq_low"].sample(ba) > -20
            gb = b.lanes["eq_low"].sample(bb) > -20
            if a.stems:
                ga &= a.lanes[STEM_LANES[stemlib.low_stem(ea)]].sample(ba) > 0.3  # the low end lives in the bass stem (or drums when bass is empty)
            if b.stems:
                gb &= b.lanes[STEM_LANES[stemlib.low_stem(eb)]].sample(bb) > 0.3
            va = a.lanes["volume"].sample(ba) > 0.3
            vb = b.lanes["volume"].sample(bb) > 0.3
            clash = ga & gb & va & vb
            if clash.mean() > 0.15:
                first, last = ts[clash][0], ts[clash][-1]
                add("warn", b.id, f"bass clash with {a.id}: both low bands open and audible for {clash.sum() * 0.25:.0f}s (set time {first:.0f}-{last:.0f}s): kill one low band (apply_transition bass_swap does this)")
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
