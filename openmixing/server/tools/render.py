from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import Context

from ... import candidates, engine, validate
from ... import plan as planlib
from ...core.errors import OpenMixingError
from ...engine import tracklist as tracklistlib
from ...model.setmodel import SetModel
from ..app import tool
from ..renders import do_render, publish
from ..state import LOCK, clip_of, cur, info, lib, span
from ..versions import versions
from ..views import jsonable


@tool()
def export_tracklist(format: Literal["text", "json", "cue"] = "text", set_name: str | None = None, offset_sec: float | None = None) -> dict:
    """Tracklist / cue sheet of the set: start time of every clip in the rendered file (time 0 = the earliest clip, or offset_sec) with
    artist, title, bpm and key from the track tags. Written to .openmixing/renders/<set>.tracklist.txt|json or <set>.cue next to the render."""
    m = lib().load_set(set_name) if set_name else cur()
    if not m.clips:
        raise ValueError("the set has no clips")
    tl = engine.timeline(m, info)
    rs = tracklistlib.rows(tl, info, lambda tid: lib().merged_tags(info(tid)["id"], lib().analyze(tid)["tags"]), offset_sec)
    audio = next((f"{lib().safe_name(m.name)}.{x}" for x in ("mp3", "flac", "wav") if lib().render_path(m.name, x).exists()),
                 f"{lib().safe_name(m.name)}.mp3")
    ext, body = {"text": ("tracklist.txt", tracklistlib.as_text(m.name, rs)), "json": ("tracklist.json", tracklistlib.as_json(m.name, rs, audio)),
                 "cue": ("cue", tracklistlib.as_cue(m.name, rs, audio))}[format]
    path = lib().output_path(lib().state / "renders" / f"{lib().safe_name(m.name)}.{ext}", "set_name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return {"path": str(path), "format": format, "tracks": len(rs), "audio_file": audio, "content": body if format != "json" else rs}


@tool()
async def render(start_sec: float | None = None, end_sec: float | None = None, format: Literal["mp3", "flac", "wav"] = "mp3",
                 master_gain_db: float = 0.0, ctx: Context = None) -> dict:
    """Render the current set (or a time window of it) to .openmixing/renders/<set>.<fmt> (default mp3 320 kbps CBR; flac / wav on request) with a
    master limiter, and report integrated loudness / loudness range / true peak. The full render is what open_player plays."""
    loop = asyncio.get_running_loop()
    def progress(i, n, msg):
        if ctx:
            asyncio.run_coroutine_threadsafe(ctx.report_progress(i, n, msg), loop)
    with LOCK:
        model = SetModel.from_dict(cur().to_dict())
    name = model.name if start_sec is None and end_sec is None else f"{model.name}.window"
    meta = await asyncio.to_thread(do_render, name, start_sec, end_sec, format, master_gain_db, progress, model)
    if start_sec is None and end_sec is None:
        await asyncio.to_thread(publish, meta, model)
    meta.pop("peaks", None)
    return jsonable(meta)


@tool()
async def preview_transition(out_clip: str, in_clip: str, bars_before: float = 8.0, bars_after: float = 8.0,
                             format: Literal["mp3", "flac", "wav"] = "mp3") -> dict:
    """Render just the overlap of two clips plus some bars of lead-in / lead-out to .openmixing/renders/ for quick checking."""
    o, i = clip_of(out_clip), clip_of(in_clip)
    (_, eo), (si, ei) = span(o), span(i)
    bar = 4 * 60.0 / info(i.track)["bpm"]
    a, b = max(si - bars_before * bar, 0.0), min(eo, ei) + bars_after * bar
    with LOCK:
        model = SetModel.from_dict(cur().to_dict())
    meta = await asyncio.to_thread(do_render, f"preview_{out_clip}_{in_clip}", a, b, format, 0.0, None, model)
    meta.pop("peaks", None)
    meta["window"] = [round(a, 2), round(b, 2)]
    return jsonable(meta)


def _window(m: SetModel) -> tuple[float, float]:
    rows = engine.timeline(m, info)
    lo, hi = max(r["start_sec"] for r in rows), min(r["end_sec"] for r in rows)
    return (lo, hi) if hi > lo else (min(r["end_sec"] for r in rows),) * 2


@tool()
async def render_variations(a: str, b: str, count: int = 3, goals: list[str] | dict[str, float] | None = None,
                            plans: list[dict] | None = None, bars_before: float = 8.0, bars_after: float = 8.0,
                            format: Literal["mp3", "flac", "wav"] = "mp3") -> dict:
    """Render several distinct takes of one transition, each to .openmixing/renders/<id>.<fmt> with its validation, so you can compare them.

    Takes come from the best `count` suggest_transitions candidates for a -> b with different styles (goals as there), or from your own `plans`
    (list of plan
    documents, e.g. edited candidates; a and b are then only used for naming). Each take renders the overlap plus bars_before / bars_after.
    The current set is not touched. Returns one compact row per take: id, style, score, window, loudness, true peak, blocking failures,
    beat drift, loudness jump, low-end overlap seconds and intent pass / fail. Use compare_renders(id1, id2) for the full diff."""
    ta, tb = lib().resolve(a), lib().resolve(b)
    takes = []
    if plans:
        takes = [{"label": f"plan{i + 1}", "plan": p, "style": p.get("name", "plan"), "bars": None, "total": None} for i,
                 p in enumerate(plans[:count])]
    else:
        sug = await asyncio.to_thread(candidates.suggest, lib(), a, b, goals, max(count * 4, 8))
        picked: list[dict] = []
        for c in sug["candidates"]:
            if c["style"] not in {x["style"] for x in picked}:
                picked.append(c)
        for c in sug["candidates"]:
            if c not in picked and (c["style"], c["bars"]) not in {(x["style"], x["bars"]) for x in picked}:
                picked.append(c)
        takes = [{"label": c["id"], "plan": c["plan"], "style": c["style"], "bars": c["bars"], "total": c["total"]} for c in picked[:count]]
    rows = []
    for i, t in enumerate(takes):
        model, errs = planlib.build_plan(t["plan"], lib())
        if errs:
            rows.append({"take": t["label"], "style": t["style"], "error": errs[:3]})
            continue
        lo, hi = _window(model)
        bar = 240.0 / info(model.clips[-1].track)["bpm"]
        w0, w1 = max(lo - bars_before * bar, 0.0), hi + bars_after * bar
        rid = f"var_{ta['id'][:6]}_{tb['id'][:6]}_{i + 1}_{t['style']}" + (f"_{t['bars']}b" if t["bars"] else "")
        meta = await asyncio.to_thread(do_render, rid, w0, w1, format, 0.0, None, model)
        rows.append({"id": rid, "take": t["label"], "style": t["style"], "bars": t["bars"], "score": t["total"],
                     "window_sec": [round(w0, 1), round(w1, 1)],
                     "path": meta["path"], **meta["metrics"]})
    return jsonable({"a": ta["file"], "b": tb["file"], "takes": rows, "versions": versions(),
                     "note": "metrics are measurements; judge them against what you intended. compare_renders(id1, id2) diffs two takes"})


def _side(ref: str) -> dict:
    name = Path(ref).name
    for ext in (".mp3", ".flac", ".wav", ".validation.json"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    p = lib().output_path(lib().state / "renders" / f"{lib().safe_name(name)}.validation.json", "ref")
    if not p.exists():
        have = sorted(x.name[: -len(".validation.json")] for x in p.parent.glob("*.validation.json"))
        raise OpenMixingError("not_found", f"no validated render {ref!r}", "ref", f"render it first; available: {have[:20]}")
    return json.loads(p.read_text())


@tool()
def compare_renders(a: str, b: str, detail: bool = False) -> dict:
    """Diff the validation metrics of two renders (ids from render_variations / render, or file names): loudness, true peak, clipping,
    unplanned silence, beat drift, loudness jump, low-end overlap, intent pass / fail. `delta` = b - a; `lower` says which render has
    the smaller value of a lower-is-better metric, it is not a verdict. detail=True adds each render's intent report."""
    sa, sb = _side(a), _side(b)
    out = {"a": sa["render"], "b": sb["render"], "diff": validate.compare(sa["metrics"], sb["metrics"]), "versions": versions()}
    if detail:
        out["intent_report"] = {"a": sa["validation"].get("intent_report", []), "b": sb["validation"].get("intent_report", [])}
    return jsonable(out)
