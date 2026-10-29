from __future__ import annotations

from ..core.errors import OpenMixingError
from ..core.units import BAR
from ..model.clip import Clip
from ..model.lane import LEGACY_STEM_NAMES, STEM_LANES
from .context import Ctx
from .envelope import parse_envelope
from .events import parse_loops
from .fx import parse_fx
from .ranges import RANGES
from .rate import parse_rate
from .schema import BANDS, ID_RE
from .techniques import parse_techniques
from .triggers import parse_triggers


def parse_src(cx: Ctx, lib, tid: str, v, path: str) -> float | None:
    try:
        return lib.to_bars(tid, v, path)
    except OpenMixingError as e:
        cx.add(e.code, path, e.message, e.fix)
    except (ValueError, KeyError) as e:
        cx.add("invalid_position", path, str(e.args[0] if e.args else e), "use bars, 'N beats', 'cue:<name>' or 'section:<label>'")
    return None


CLIP_KEYS = ("id", "track", "label", "src_start", "bars", "start", "bpm", "tempo", "gain_db", "semitones", "key_lock", "stems",
             "pitch", "key_match", "eq", "filter", "volume", "stem_levels", "fx", "loops", "rate", "triggers", "techniques")


def parse_clip(cx: Ctx, lib, c, i: int, ids: set[str], seq: int) -> Clip | None:
    p = f"clips[{i}]"
    if not cx.obj(c, p, CLIP_KEYS, ("track",)):
        return None
    cid = c.get("id", f"c{seq}")
    if not isinstance(cid, str) or not ID_RE.match(cid):
        cx.add("invalid_id", f"{p}.id", f"bad clip id {cid!r}", "use letters, digits, _ or - (max 40)")
        cid = f"c{seq}"
    if cid in ids:
        cx.add("duplicate_id", f"{p}.id", f"clip id {cid!r} is used twice", "give each clip a unique id")
    ids.add(cid)
    try:
        e = lib.effective(c["track"])
    except (KeyError, ValueError) as ex:
        cx.add("track_not_found", f"{p}.track", str(ex.args[0] if ex.args else ex), "call list_tracks and use an id or file name")
        return None
    src = parse_src(cx, lib, e["id"], c.get("src_start", 0.0), f"{p}.src_start")
    n_bars = e["n_bars"]
    length = None
    if "bars" in c:
        length = parse_src(cx, lib, e["id"], c["bars"], f"{p}.bars")
    elif src is not None:
        length = n_bars - src
    if src is not None and src >= n_bars:
        cx.add("out_of_range", f"{p}.src_start", f"{src:g} is past the end of the track ({n_bars} bars)", f"use a bar below {n_bars}")
        src = None
    if length is not None and length <= 0:
        cx.add("out_of_range", f"{p}.bars", "the clip must play for more than 0 bars", "give a positive length")
        length = None
    clip = Clip(cid, e["id"], (src or 0.0) * BAR, (length or 1.0) * BAR, label=str(c.get("label") or e["file"].rsplit(".", 1)[0]))
    L = length if length is not None else 1.0
    cx.clips[cid], cx.cur = (e["id"], (src or 0.0)), cid
    for key, attr in (("gain_db", "gain_db"), ("semitones", "semitones")):
        if key in c and (v := cx.num(c[key], f"{p}.{key}", key=key)) is not None:
            setattr(clip, attr, v)
    if "key_lock" in c:
        if isinstance(c["key_lock"], bool):
            clip.key_lock = c["key_lock"]
        else:
            cx.add("wrong_type", f"{p}.key_lock", "expected true or false")
    if "stems" in c:
        if not isinstance(c["stems"], bool):
            cx.add("wrong_type", f"{p}.stems", "expected true or false")
        elif c["stems"] and not e.get("stems"):
            cx.add("missing_stems", f"{p}.stems", f"{e['file']} has no stems yet" if not e.get("stems_legacy")
                   else f"{e['file']} only has the old 3-stem layout",
                   "call separate_stems" + (" with force=True" if e.get("stems_legacy") else " first"))
        else:
            clip.stems = c["stems"]
    if "bpm" in c and (v := cx.num(c["bpm"], f"{p}.bpm", key="bpm")) is not None:
        clip.tempo.points = [[0.0, v]]
    if "tempo" in c:
        if not clip.tempo.points:
            clip.tempo.points = [[0.0, float(e["bpm"])]]
        parse_envelope(cx, c["tempo"], f"{p}.tempo", clip.tempo, "bpm", *RANGES["bpm"], L)
    if "pitch" in c:
        parse_envelope(cx, c["pitch"], f"{p}.pitch", clip.pitch, "semitones", *RANGES["semitones"], L)
    if "eq" in c and cx.obj(c["eq"], f"{p}.eq", BANDS):
        for b in BANDS:
            if b in c["eq"]:
                parse_envelope(cx, c["eq"][b], f"{p}.eq.{b}", clip.lanes[f"eq_{b}"], "gain_db", *RANGES["eq"], L, kill=True)
    for key, vk, rk in (("filter", "value", "filter"), ("volume", "level", "volume")):
        if key in c:
            parse_envelope(cx, c[key], f"{p}.{key}", clip.lanes[key], vk, *RANGES[rk], L)
    if isinstance(c.get("stem_levels"), dict):
        sl = dict(c["stem_levels"])
        for old, new in LEGACY_STEM_NAMES.items():
            if old in sl:
                for n in new:
                    sl.setdefault(n, sl[old])
                del sl[old]
        c = {**c, "stem_levels": sl}
    if "stem_levels" in c and cx.obj(c["stem_levels"], f"{p}.stem_levels", tuple(STEM_LANES)):
        if not clip.stems and c["stem_levels"]:
            cx.add("stems_off", f"{p}.stem_levels", "stem levels need stems=true on this clip", "set \"stems\": true (after separate_stems)")
        for s, lane in STEM_LANES.items():
            if s in c["stem_levels"]:
                parse_envelope(cx, c["stem_levels"][s], f"{p}.stem_levels.{s}", clip.lanes[lane], "level", *RANGES["stem"], L)
    parse_fx(cx, c.get("fx", []), p, clip, L)
    parse_loops(cx, c.get("loops", []), p, clip, L)
    parse_rate(cx, c.get("rate", []), p, clip, L)
    parse_triggers(cx, c.get("triggers", []), p, clip, L)
    parse_techniques(cx, c.get("techniques", []), p, clip, L,
                     lib.analyze(e["id"])["structure"]["sections"] if c.get("techniques") else [])
    return clip


def parse_start(cx: Ctx, c, i: int, clip: Clip, prev: Clip | None, ids_all: set[str]) -> None:
    p = f"clips[{i}].start"
    s = c.get("start")
    if s is None:
        clip.start = {"anchor": prev.id, "anchor_beat": prev.length_beats} if prev else {"at_sec": 0.0}
        return
    if not isinstance(s, dict):
        cx.add("wrong_type", p, "expected an object")
        return
    if "over" in s:
        if not cx.obj(s, p, ("over", "at_bar", "offset_ms"), ("over", "at_bar")):
            return
        if s["over"] not in ids_all:
            cx.add("unknown_clip", f"{p}.over", f"no clip {s['over']!r} in this plan", f"clip ids: {sorted(ids_all)}")
            return
        if s["over"] == clip.id:
            cx.add("anchor_cycle", f"{p}.over", "a clip cannot be anchored to itself", "anchor it to another clip")
            return
        at = cx.out_bars(s["at_bar"], f"{p}.at_bar", s["over"])
        if at is not None:
            clip.start = {"anchor": s["over"], "anchor_beat": at * BAR}
    else:
        if not cx.obj(s, p, ("at_sec", "offset_ms"), ("at_sec",)):
            return
        at = cx.num(s["at_sec"], f"{p}.at_sec", 0.0, None)
        if at is not None:
            clip.start = {"at_sec": at}
    if "offset_ms" in s and (o := cx.num(s["offset_ms"], f"{p}.offset_ms", key="offset_ms")) is not None:
        clip.start["offset_ms"] = o
