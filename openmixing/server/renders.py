from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import engine, repro, validate
from ..core.jsonio import write_json
from ..model.setmodel import SetModel
from ..player import build_state
from .state import cur, info, lib
from .versions import versions
from .views import jsonable

CEILING_DB = -1.5


def qa(meta: dict) -> dict:
    notes = []
    if meta.get("true_peak_dbtp") is not None and meta["true_peak_dbtp"] > -0.5:
        notes.append("true peak above -0.5 dBTP: lower master_gain_db")
    if meta.get("integrated_lufs") is not None and meta["integrated_lufs"] < -20:
        notes.append("quiet render; raise master_gain_db or check clip volumes")
    pk = np.array(meta.get("peaks") or [0])
    return {"notes": notes, "max_block_peak": float(pk.max())}


def render_params(start_sec, end_sec, fmt, master_gain_db) -> dict:
    return {"format": fmt, "master_gain_db": master_gain_db, "ceiling_db": CEILING_DB, "start_sec": start_sec, "end_sec": end_sec}


def do_render(name: str, start_sec, end_sec, fmt, master_gain_db, progress=None, model: SetModel | None = None, cancel=None) -> dict:
    m = model or cur()
    path = lib().render_path(name, fmt)
    meta = engine.render_set(m, info, path, start_sec, end_sec, fmt, master_gain_db, CEILING_DB, progress, cancel)
    meta["set"] = m.name
    meta["qa"] = qa(meta)
    meta["versions"] = versions()
    meta["validation"] = validate.validate_render(path, m, info, [r for r in meta["clips"]], meta["start_sec"], meta)
    meta.pop("low_db", None)
    meta.pop("meters", None)
    meta["metrics"] = validate.summarize(meta["validation"], meta)
    side = {"render": path.name, "set": m.name, "metrics": meta["metrics"], "validation": meta["validation"],
            "window": [meta["start_sec"], meta["end_sec"]]}
    write_json(path.with_suffix(".validation.json"), jsonable(side))
    repro.write_record(path, jsonable(repro.build_record(m, meta, render_params(start_sec, end_sec, fmt, master_gain_db), versions(), lib(), info)))
    return meta


def publish(meta: dict, model: SetModel | None = None) -> None:
    side = Path(meta["path"]).with_suffix(".json")
    write_json(side, jsonable(build_state(lib(), model or cur(), meta)))
