from __future__ import annotations

import json
from hashlib import blake2b
from pathlib import Path
from typing import Callable

from ..analysis import stems as stemlib
from ..dsp.synth import seed_of
from ..library import ANALYSIS_VERSION, Library
from ..model.setmodel import SetModel


def track_ids(model: SetModel) -> list[str]:
    return sorted({c.track for c in model.clips})


def tracks(model: SetModel, lib: Library, info: Callable[[str], dict]) -> list[dict]:
    out = []
    for ref in track_ids(model):
        e = info(ref)
        out.append({"ref": ref, "id": e["id"], "file": e["file"], "blake2b": e["id"], "analysis_version": ANALYSIS_VERSION,
                    "overrides": e["overrides"], "cues": lib.cues(e["id"]), "gain_db": e["gain_db"]})
    return out


def samples(model: SetModel) -> list[dict]:
    return [{"slot": k, "path": s.path, "blake2b": Library.content_id(s.path)} for k, s in sorted(model.sampler.items())]


def seeds(model: SetModel) -> dict[str, int]:
    return {t.id: seed_of(t.id) for c in model.clips for t in c.triggers if t.kind != "sample"}


def stems(model: SetModel, info: Callable[[str], dict], full: bool) -> dict:
    h = {}
    for ref in sorted({c.track for c in model.clips if c.stems}):
        paths = info(ref)["stems"] or {}
        h[ref] = {n: Library.content_id(p) if full else f"{Path(p).stat().st_size}:{Path(p).stat().st_mtime_ns}" for n, p in sorted(paths.items())}
    return {"model": stemlib.MODEL, "cache_version": stemlib.CACHE_VERSION, "hashes": h}


def fingerprint(model: SetModel, params: dict, versions: dict, lib: Library, info: Callable[[str], dict]) -> str:
    ts = [(t["id"], t["overrides"], t["gain_db"]) for t in tracks(model, lib, info)]
    body = {"set": model.to_dict(), "params": params, "versions": versions, "tracks": ts,
            "samples": samples(model), "stems": stems(model, info, False)}
    return blake2b(json.dumps(body, sort_keys=True, default=str).encode(), digest_size=16).hexdigest()
