import json
import re
import threading
import time
from pathlib import Path

import pytest
from synth import make_track

import openmixing
from openmixing import engine
from openmixing import server as S
from openmixing.core.cancel import JobCancelled
from openmixing.core.errors import OpenMixingError
from openmixing.jobs import JobManager
from openmixing.library import ANALYSIS_VERSION, Library
from openmixing.model import SetModel


@pytest.fixture
def folder(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    make_track(tmp_path / "b.wav", bpm=120, bars=16, pad_hz=392)
    S.STATE.update(lib=Library(tmp_path), set=None, jobs=None)
    for n in ("a", "b"):
        S.override_track(f"{n}.wav", bpm=120.0, first_downbeat_sec=0.0)
    S.new_set("j")
    ca = S.add_clip("a.wav", bars=4)["clip"]
    S.add_clip("b.wav", bars=4, over_clip=ca, at_bar=2)
    S.add_trigger(ca, "riser", 1, 1)
    yield tmp_path
    S.STATE.update(lib=None, set=None, jobs=None)


def finish(r):
    return S.jobs().wait(r["id"], 120)


def test_render_twice_identical_hash_and_record(folder):
    a = S.start_render(idempotency_key="1")
    j = finish(a)
    first = j.result["blake2b"]
    b = S.start_render(idempotency_key="2")
    assert not b["reused"]
    assert finish(b).result["blake2b"] == first
    rec = S.lib().output_path(Path(j.result["path"]).with_suffix(".repro.json"))
    r = json.loads(rec.read_text())
    assert r["output"]["blake2b"] == first and r["params"]["mp3_bitrate_kbps"] == 320 and r["params"]["format"] == "mp3"
    assert re.match(r"ffmpeg version", r["ffmpeg"]) and r["libs"]["numpy"] and r["seeds"]
    assert r["versions"]["analysis"] == ANALYSIS_VERSION and r["tracks"][0]["blake2b"] and r["set"]["name"] == "j"


def test_same_key_runs_once(folder, monkeypatch):
    calls = []
    real = engine.render_set
    monkeypatch.setattr(engine, "render_set", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = S.start_render()
    b = S.start_render()
    assert b["reused"] and b["id"] == a["id"]
    finish(a)
    c = S.start_render()
    assert c["reused"] and c["id"] == a["id"] and c["state"] == "done"
    assert len(calls) == 1
    S.add_trigger(S.cur().clips[0].id, "impact", 3, 1)
    d = S.start_render()
    assert not d["reused"] and d["id"] != a["id"]
    finish(d)
    assert len(calls) == 2


def test_modified_output_is_not_reused(folder):
    a = S.start_render()
    j = finish(a)
    Path(j.result["path"]).write_bytes(b"x")
    assert not S.start_render()["reused"]


def test_cancel_leaves_no_temp_files(folder, tmp_path):
    out = tmp_path / ".openmixing" / "renders" / "c.mp3"
    seen = []

    def progress(i, n, msg):
        seen.append(i)

    with pytest.raises(JobCancelled):
        engine.render_set(S.cur(), S.info, out, progress=progress, cancel=lambda: bool(seen))
    assert list(out.parent.glob("c.*")) == []


def test_cancel_during_limiter_block_cleans_up(folder, tmp_path):
    out = tmp_path / ".openmixing" / "renders" / "c.flac"
    calls = []
    with pytest.raises(JobCancelled):
        engine.render_set(S.cur(), S.info, out, fmt="flac", cancel=lambda: calls.append(1) is None and len(calls) == len(S.cur().clips) + 1)
    assert list(out.parent.glob("c.*")) == []


def test_manager_cancel_running_and_queued(tmp_path):
    m = JobManager(tmp_path / "jobs.jsonl")
    go = threading.Event()

    def slow(job):
        go.set()
        while not job.cancelled():
            time.sleep(0.01)
        raise JobCancelled()

    a, _ = m.submit("render", "k1", slow)
    b, _ = m.submit("render", "k2", lambda j: {"ok": 1})
    go.wait(5)
    m.cancel(b.id)
    m.cancel(a.id)
    m.wait(a.id, 5)
    assert (a.state, b.state) == ("cancelled", "cancelled")
    assert m.submit("render", "k1", lambda j: {"x": 1})[1] is False


def test_failed_job_reports_error(tmp_path):
    m = JobManager(tmp_path / "jobs.jsonl")
    j, _ = m.submit("render", "k", lambda j: (_ for _ in ()).throw(ValueError("boom")))
    m.wait(j.id, 5)
    assert j.state == "failed" and j.error["message"] == "boom"


def test_jobs_survive_restart_and_running_is_lost(tmp_path):
    p = tmp_path / "jobs.jsonl"
    m = JobManager(p)
    done, _ = m.submit("render", "k", lambda j: {"v": 1})
    m.wait(done.id, 5)
    go = threading.Event()
    stuck, _ = m.submit("render", "k2", lambda j: go.wait(5) and {})
    time.sleep(0.2)
    m2 = JobManager(p)
    assert m2.get(done.id).state == "done" and m2.get(done.id).result == {"v": 1}
    assert m2.get(stuck.id).state == "lost"
    go.set()


def test_verify_render_metrics_and_bit_exact(folder):
    j = finish(S.start_render())
    name = Path(j.result["path"]).name
    r = S.verify_render(name)
    assert r["ok"] and r["level"] == "metrics" and abs(r["delta"]["lufs"]) <= 0.1
    assert S.verify_render(name, "bit_exact")["identical_bytes"]
    assert not list(Path(j.result["path"]).parent.glob("*.verify.*"))


def test_verify_render_detects_modified_track(folder):
    j = finish(S.start_render())
    make_track(folder / "a.wav", bpm=120, bars=16, pad_hz=500)
    with pytest.raises(OpenMixingError) as e:
        S.verify_render(Path(j.result["path"]).name)
    assert e.value.code == "hash_mismatch" and re.fullmatch(r"tracks\[\d+\]\.blake2b", e.value.field)


def test_verify_without_record(folder):
    with pytest.raises(OpenMixingError):
        S.verify_render("nothing")


def test_capabilities_and_versions(folder):
    c = S.capabilities()
    assert c["jobs"]["supported_kinds"] == ["render"] and c["jobs"]["max_concurrent"] == 1
    assert c["reproducibility"]["default_verify"] == "metrics"
    assert (c["versions"]["analysis"], c["versions"]["plan_schema"], c["versions"]["validation"]) == (ANALYSIS_VERSION, "1.1", 3)
    for k in ("rate", "dynamics", "generators", "sampler", "techniques", "set_builder", "library", "validation", "plan_ranges"):
        assert c[k]


def test_list_jobs_filters(folder):
    finish(S.start_render())
    assert len(S.list_jobs("done")["jobs"]) == 1 and S.list_jobs("failed")["jobs"] == []


def test_dsp_never_uses_unseeded_random():
    bad = []
    for p in (Path(openmixing.__file__).parent / "dsp").rglob("*.py"):
        for ln in p.read_text().splitlines():
            if re.search(r"np\.random\.(?!default_rng\([^)]|Generator)", ln) or re.search(r"(?<!np\.)\brandom\.", ln):
                bad.append((p.name, ln))
    assert not bad


def test_set_model_snapshot_independent(folder):
    snap = SetModel.from_dict(S.cur().to_dict())
    S.cur().clips.pop()
    assert len(snap.clips) == 2
