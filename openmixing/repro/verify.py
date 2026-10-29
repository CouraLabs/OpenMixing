from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..core.errors import OpenMixingError
from ..core.jsonio import read_json
from ..engine import render_set
from ..library import Library
from ..model.setmodel import SetModel
from .record import environment, file_hash, record_path

TOLERANCE = {"lufs": 0.1, "true_peak": 0.1, "duration": 0.05}


def _check_files(rec: dict, lib: Library) -> None:
    for key, rows, field in (("tracks", rec["tracks"], "file"), ("samples", rec["samples"], "path")):
        for i, r in enumerate(rows):
            p = lib.root / r["file"] if field == "file" else Path(r["path"])
            if not p.exists() or Library.content_id(p) != r["blake2b"]:
                raise OpenMixingError("hash_mismatch", f"{key[:-1]} {r[field]!r} changed since the render was recorded", f"{key}[{i}].blake2b",
                                      "restore the original file; the record cannot be reproduced without it")


def _differs(rec: dict, now: dict) -> dict:
    flat = lambda e: {**{f"libs.{k}": v for k, v in e["libs"].items()}, **{f"versions.{k}": v for k, v in e["versions"].items()},
                      "ffmpeg": e["ffmpeg"], "python": e["python"], "platform": e["platform"]}
    a, b = flat(rec), flat(now)
    return {k: {"recorded": a.get(k), "now": b.get(k)} for k in sorted(a | b) if a.get(k) != b.get(k)}


def verify(render: Path, level: str, lib: Library, info: Callable[[str], dict], versions: dict) -> dict:
    rec = read_json(record_path(render))
    if rec is None:
        raise OpenMixingError("not_found", f"no reproducibility record for {render.name}", "render_name", "render the set first")
    _check_files(rec, lib)
    p, out = rec["params"], rec["output"]
    tmp = render.with_name(f"{render.stem}.verify{render.suffix}")
    try:
        meta = render_set(SetModel.from_dict(rec["set"]), info, tmp, p["start_sec"], p["end_sec"], p["format"], p["master_gain_db"], p["ceiling_db"])
        got = {"blake2b": file_hash(tmp), "duration": meta["duration_sec"], "lufs": meta["integrated_lufs"], "true_peak": meta["true_peak_dbtp"]}
    finally:
        tmp.unlink(missing_ok=True)
    delta = {k: round(got[k] - out[k], 3) for k in TOLERANCE if got[k] is not None and out[k] is not None}
    ok = got["blake2b"] == out["blake2b"] if level == "bit_exact" else all(abs(delta[k]) <= TOLERANCE[k] for k in delta)
    return {"ok": ok, "level": level, "identical_bytes": got["blake2b"] == out["blake2b"], "delta": delta, "tolerance": TOLERANCE,
            "recorded": out, "got": got, "environment_differences": {} if ok else _differs(rec, environment(versions))}
