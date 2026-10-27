"""Local web UI backend: pick a folder -> analyse -> order -> render the mix -> play in the browser."""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..analysis.analyze import (AUDIO_EXT, analyze_file, cache_key, load_cache, save_cache, scan_folder)
from ..decode import decode
from ..models import SR, Track
from ..planner.rules import order_tracks, plan_transition
from ..render.mix import render_set

WORK = Path.home() / ".openmixing"
WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="OpenMixing")
lock = threading.Lock()
state: dict = {
    "folder": None, "tracks": [], "analysis": {"running": False, "done": 0, "total": 0, "errors": []},
    "mix": {"running": False, "done": 0, "total": 0, "error": None, "result": None, "version": 0},
}


def _meta(t: dict) -> Track:
    return Track(t["id"], None, t["bpm"], t["first_downbeat"], 4, t["grid_stable"], t["gain_db"], t["length"])


# ---------------------------------------------------------------- folder browsing
@app.get("/api/browse")
def browse(path: str = ""):
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        raise HTTPException(404, f"not a folder: {p}")
    dirs = []
    try:
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.is_dir() and not c.name.startswith("."):
                dirs.append(c.name)
    except PermissionError:
        pass
    n_audio = sum(1 for c in p.iterdir() if c.suffix.lower() in AUDIO_EXT) if os.access(p, os.R_OK) else 0
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "audio_files": n_audio}


@app.post("/api/pick-folder")
def pick_folder():
    """Native folder chooser (macOS)."""
    if sys.platform != "darwin":
        raise HTTPException(501, "native picker only on macOS; type the path instead")
    r = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder with prompt "Choose your music folder")'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"path": None}  # cancelled
    return {"path": r.stdout.strip()}


# ---------------------------------------------------------------- analysis
class ScanReq(BaseModel):
    path: str


def _run_analysis(files: list[str]) -> None:
    cache = load_cache()
    results: dict[str, dict] = {}
    todo = []
    for f in files:
        try:
            hit = cache.get(cache_key(f))
        except OSError:
            continue
        if hit:
            results[f] = hit
        else:
            todo.append(f)
    with lock:
        state["analysis"].update(done=len(results), total=len(files), errors=[])
        state["tracks"] = [results[f] for f in files if f in results]
    if todo:
        with ProcessPoolExecutor(max(1, min(4, (os.cpu_count() or 2) - 1)), mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(analyze_file, f): f for f in todo}
            for fut in as_completed(futs):
                f = futs[fut]
                try:
                    results[f] = fut.result()
                    cache[cache_key(f)] = results[f]
                except Exception as e:  # unreadable/corrupt file: report, keep going
                    with lock:
                        state["analysis"]["errors"].append(f"{Path(f).name}: {e}")
                with lock:
                    state["analysis"]["done"] += 1
                    state["tracks"] = [results[x] for x in files if x in results]
        save_cache(cache)
    with lock:
        # a good default order straight away; the user can reorder
        state["tracks"] = order_tracks([results[f] for f in files if f in results])
        state["analysis"]["running"] = False


@app.post("/api/scan")
def scan(req: ScanReq):
    folder = Path(req.path).expanduser()
    if not folder.is_dir():
        raise HTTPException(404, f"not a folder: {folder}")
    files = scan_folder(str(folder))
    if not files:
        raise HTTPException(400, "no audio files found in that folder")
    with lock:
        if state["analysis"]["running"] or state["mix"]["running"]:
            raise HTTPException(409, "busy")
        state.update(folder=str(folder), tracks=[])
        state["analysis"] = {"running": True, "done": 0, "total": len(files), "errors": []}
        state["mix"].update(result=None)
    threading.Thread(target=_run_analysis, args=(files,), daemon=True).start()
    return {"files": len(files)}


@app.get("/api/state")
def get_state():
    with lock:
        return {"folder": state["folder"], "tracks": state["tracks"], "analysis": state["analysis"],
                "mix": state["mix"]}


class TrackEdit(BaseModel):
    bpm: float | None = None
    first_downbeat: float | None = None


@app.post("/api/track/{tid}")
def edit_track(tid: str, edit: TrackEdit):
    """Manual correction of the beat grid (analysis can be wrong, especially tempo octave / downbeat)."""
    with lock:
        t = next((t for t in state["tracks"] if t["id"] == tid), None)
        if not t:
            raise HTTPException(404)
        if edit.bpm is not None:
            if not 40 <= edit.bpm <= 220:
                raise HTTPException(400, "bpm must be 40..220")
            t["bpm"] = edit.bpm
        if edit.first_downbeat is not None:
            t["first_downbeat"] = max(edit.first_downbeat, 0.0)
        t["grid_stable"] = True  # a hand-set grid is trusted
        cache = load_cache()
        cache[cache_key(t["path"])] = t
        save_cache(cache)
    return t


class OrderReq(BaseModel):
    ids: list[str] | None = None  # None = auto-order everything


@app.post("/api/order")
def set_order(req: OrderReq):
    with lock:
        if req.ids is None:
            state["tracks"] = order_tracks(state["tracks"])
        else:
            by_id = {t["id"]: t for t in state["tracks"]}
            if set(req.ids) != set(by_id):
                raise HTTPException(400, "ids must be a permutation of the current tracks")
            state["tracks"] = [by_id[i] for i in req.ids]
        return state["tracks"]


# ---------------------------------------------------------------- mixing
def _run_mix(tracks: list[dict], version: int) -> None:
    out = WORK / f"mix_{version}.flac"
    try:
        metas = [_meta(t) for t in tracks]
        plans, notes = [], []
        min_from = 0
        for i in range(len(metas) - 1):
            plan = plan_transition(metas[i], metas[i + 1], min_from)
            plans.append(plan)
            if plan and not plan.beatmatch:
                min_from = plan.to_bar + plan.length_bars + 2
                notes.append({"kind": "echo", "bars": plan.length_bars})
            elif plan:
                min_from = plan.to_bar + plan.length_bars + 2
                notes.append({"kind": "blend", "bars": plan.length_bars, "from_bpm": metas[i].bpm,
                              "to_bpm": metas[i + 1].bpm})
            else:
                min_from = 0
                notes.append({"kind": "cut"})
        by_id = {t["id"]: t for t in tracks}

        def load(meta: Track) -> Track:
            return Track(meta.id, decode(by_id[meta.id]["path"]), meta.bpm, meta.first_downbeat, 4,
                         meta.grid_stable, meta.gain_db)

        def progress(done, total):
            with lock:
                state["mix"].update(done=done, total=total)

        timeline = render_set(metas, plans, load, str(out), progress)
        for ev in timeline:
            t = tracks[ev["index"]]
            ev.update(id=t["id"], name=t["name"], bpm=t["bpm"], key=t["key"],
                      transition_in=notes[ev["index"] - 1] if ev["index"] else None)
        with lock:
            state["mix"].update(running=False, error=None,
                                result={"url": f"/api/mix.flac?v={version}", "timeline": timeline,
                                        "duration": float(__import__("soundfile").info(str(out)).duration)})
    except Exception as e:
        with lock:
            state["mix"].update(running=False, error=f"{type(e).__name__}: {e}")


@app.post("/api/mix")
def build_mix():
    with lock:
        if state["analysis"]["running"] or state["mix"]["running"]:
            raise HTTPException(409, "busy")
        tracks = list(state["tracks"])
        if len(tracks) < 2:
            raise HTTPException(400, "need at least 2 analysed tracks")
        version = state["mix"]["version"] + 1
        state["mix"].update(running=True, done=0, total=len(tracks) - 1, error=None, result=None, version=version)
    WORK.mkdir(parents=True, exist_ok=True)
    for old in WORK.glob("mix_*.flac"):
        try:
            old.unlink()
        except OSError:
            pass
    threading.Thread(target=_run_mix, args=(tracks, version), daemon=True).start()
    return {"ok": True}


@app.get("/api/mix.flac")
def mix_file(v: int = 0):
    path = WORK / f"mix_{state['mix']['version']}.flac"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/flac")


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    print(f"OpenMixing → http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
