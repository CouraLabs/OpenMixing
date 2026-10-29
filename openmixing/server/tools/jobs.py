from __future__ import annotations

from pathlib import Path
from typing import Literal

from ... import repro
from ...jobs import KINDS, MAX_CONCURRENT, STATES, Job
from ...model.setmodel import SetModel
from ..app import tool
from ..renders import do_render, publish, render_params
from ..state import LOCK, cur, info, jobs, lib
from ..versions import versions
from ..views import jsonable


def view(j: Job, reused: bool | None = None, result: bool = False) -> dict:
    d = {k: v for k, v in j.to_dict().items() if result or k != "result"}
    if reused is not None:
        d["reused"] = reused
    return jsonable(d)


def intact(result: dict) -> bool:
    p = Path(result["path"])
    return p.exists() and repro.file_hash(p) == result["blake2b"]


@tool()
def start_render(start_sec: float | None = None, end_sec: float | None = None, format: Literal["mp3", "flac", "wav"] = "mp3",
                 master_gain_db: float = 0.0, idempotency_key: str | None = None) -> dict:
    """Render a snapshot of the current set in the background and return {job_id, state, reused}. Poll job_status, read job_result, stop it with
    cancel_job (cooperative: it stops per clip and per 60 s limiter block and leaves no temp files). Idempotent: the default key hashes the set, the
    render parameters, library versions, track / sample / stem content; if a done job with that key still has its unchanged output, or a job is
    still running, that job is returned with reused=true and nothing is rendered twice. Pass idempotency_key to choose your own key."""
    with LOCK:
        model = SetModel.from_dict(cur().to_dict())
    params = render_params(start_sec, end_sec, format, master_gain_db)
    key = idempotency_key or repro.fingerprint(model, params, versions(), lib(), info)
    name = model.name if start_sec is None and end_sec is None else f"{model.name}.window"

    def run(job: Job) -> dict:
        def progress(i, n, msg):
            job.progress = {"i": i, "n": n, "msg": msg}
        meta = do_render(name, start_sec, end_sec, format, master_gain_db, progress, model, job.cancelled)
        if start_sec is None and end_sec is None:
            publish(meta, model)
        path = Path(meta["path"])
        return jsonable({"path": meta["path"], "format": format, "set": model.name, "blake2b": repro.file_hash(path),
                         "duration_sec": meta["duration_sec"], "integrated_lufs": meta["integrated_lufs"], "true_peak_dbtp": meta["true_peak_dbtp"],
                         "metrics": meta["metrics"], "qa": meta["qa"], "repro": str(repro.record_path(path))})

    j, reused = jobs().submit("render", key, run, intact)
    return view(j, reused)


@tool()
def job_status(job_id: str) -> dict:
    """State (queued, running, done, failed, cancelled, lost after a restart), progress {i, n, msg}, timestamps and error of one job."""
    return view(jobs().get(job_id))


@tool()
def job_result(job_id: str) -> dict:
    """The result of a done job (for a render: path, hash, loudness, metrics, repro record path); the error for a failed one."""
    return view(jobs().get(job_id), result=True)


@tool()
def list_jobs(state: Literal[*STATES] | None = None) -> dict:
    """Jobs of this folder (kept in .openmixing/jobs.jsonl), optionally only those in one state."""
    return {"kinds": list(KINDS), "max_concurrent": MAX_CONCURRENT, "jobs": [view(j) for j in jobs().list(state)]}


@tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued job at once, or ask a running one to stop; its state becomes cancelled when it reaches the next check."""
    return view(jobs().cancel(job_id))


@tool()
def verify_render(render_name: str, level: Literal["metrics", "bit_exact"] = "metrics") -> dict:
    """Re-render a render from its <render>.repro.json record (never from the live set) and compare. level="metrics" (default) accepts loudness /
    true peak / duration within tolerance; "bit_exact" compares the output hash, which only holds on the same platform and library versions.
    Fails with hash_mismatch (field tracks[i].blake2b) when a track file changed; a failed comparison lists the library versions that differ."""
    name = Path(render_name).name
    for ext in (".repro.json", ".mp3", ".flac", ".wav"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    base = lib().state / "renders" / lib().safe_name(name)
    path = next((base.with_suffix(e) for e in (".mp3", ".flac", ".wav") if repro.record_path(base.with_suffix(e)).exists()), base.with_suffix(".mp3"))
    return jsonable(repro.verify(lib().output_path(path, "render_name"), level, lib(), info, versions()))
