import json
import urllib.error
import urllib.request

import pytest
from synth import make_track

from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.jsonio import read_json, write_json
from openmixing.jobs import JobManager
from openmixing.library import Library
from openmixing.model import SetModel
from openmixing.player import start


@pytest.fixture
def folder(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    make_track(tmp_path / "b.wav", bpm=120, bars=16, pad_hz=392)
    S.STATE.update(lib=Library(tmp_path), set=None, jobs=None)
    for n in ("a", "b"):
        S.override_track(f"{n}.wav", bpm=120.0, first_downbeat_sec=0.0)
    S.new_set("r")
    ca = S.add_clip("a.wav", bars=8)["clip"]
    cb = S.add_clip("b.wav", bars=8, over_clip=ca, at_bar=4)["clip"]
    yield tmp_path, ca, cb
    S.STATE.update(lib=None, set=None, jobs=None)


def reloaded():
    return SetModel.load(S.lib().set_path(S.cur().name)).to_dict()


def code(fn, *a, **k):
    with pytest.raises(Exception) as ei:
        fn(*a, **k)
    return getattr(ei.value, "code", None) or json.loads(str(ei.value))["error"]["code"]


def test_write_json_unique_tmp_and_strict_read(tmp_path):
    f = tmp_path / "x.json"
    write_json(f, {"a": 1})
    assert read_json(f) == {"a": 1} and [p.name for p in tmp_path.iterdir()] == ["x.json"]
    f.write_text("{trunc")
    assert read_json(f, {}) == {}
    with pytest.raises(OpenMixingError) as ei:
        read_json(f, {}, strict=True)
    assert ei.value.code == "io_error"
    assert read_json(tmp_path / "none.json", 7, strict=True) == 7


def test_corrupt_overrides_fail_loudly(folder):
    (S.lib().state / "overrides.json").write_text("{trunc")
    with pytest.raises(OpenMixingError):
        S.lib().overrides()


def test_job_log_with_torn_line_loads(tmp_path):
    p = tmp_path / "jobs.jsonl"
    p.write_text('{"id": "x"\n')
    assert JobManager(p).jobs == {}


def test_edits_are_saved_to_disk(folder):
    _, ca, cb = folder
    S.set_eq(ca, "low", 2.0, "kill")
    S.add_fx(ca, "echo", 1.0, 2.0)
    S.add_loop(ca, 3.0, 1, bars=1)
    d = reloaded()
    c = next(c for c in d["clips"] if c["id"] == ca)
    assert c["lanes"]["eq_low"] and c["fx"] and c["loops"]
    S.remove_clip(cb)
    assert [c["id"] for c in reloaded()["clips"]] == [ca]


def test_out_of_range_positions_rejected(folder):
    _, ca, _ = folder
    assert code(S.set_eq, ca, "low", 40.0, "kill") == "out_of_range"
    assert code(S.set_eq, ca, "low", -1.0, "kill") == "out_of_range"
    assert code(S.add_fx, ca, "echo", 40.0, 2.0) == "out_of_range"
    S.set_eq(ca, "low", 8.0, "kill")


def test_remove_unknown_ids_not_found(folder):
    _, ca, _ = folder
    assert code(S.remove_fx, ca, "fx_nope") == "not_found"
    assert code(S.remove_loop, ca, "lp_nope") == "not_found"
    assert code(S.remove_rate, ca, "rt_nope") == "not_found"
    assert code(S.remove_trigger, ca, "tg_nope") == "not_found"
    assert code(S.remove_duck, "dk_nope") == "not_found"
    fid = S.add_fx(ca, "echo", 1.0, 2.0)["fx_id"]
    assert S.remove_fx(ca, fid)["remaining_fx"] == []


def test_cue_range_and_delete(folder):
    assert code(S.set_cue, "a.wav", "far", 1000) == "out_of_range"
    S.set_cue("a.wav", "ok", 4)
    assert code(S.delete_cue, "a.wav", "nope") == "not_found"
    assert S.delete_cue("a.wav", "ok")["cues"] == {}


def test_clear_automation_inverted_range(folder):
    _, ca, _ = folder
    assert code(S.clear_automation, ca, "all", 6.0, 2.0) == "invalid_range"


def test_player_endpoints(folder):
    tmp, _, _ = folder
    r = S.lib().state / "renders"
    r.mkdir(parents=True, exist_ok=True)
    (r / "s.json").write_text('{"ok": 1}')
    (r / "s.flac").write_bytes(bytes(range(100)))
    url = start(S.lib(), 18765)["url"].rstrip("/")
    get = lambda p, h=None: urllib.request.urlopen(urllib.request.Request(url + p, headers=h or {}))
    assert json.load(get("/api/sets")) == ["s"] and json.load(get("/api/state")) == {"ok": 1}
    assert b"<html" in get("/").read().lower()
    a = get("/audio/s.flac", {"Range": "bytes=10-19"})
    assert a.status == 206 and a.read() == bytes(range(10, 20)) and a.headers["Content-Range"] == "bytes 10-19/100"
    assert get("/audio/s.flac", {"Range": "bytes=-5"}).read() == bytes(range(95, 100))
    assert get("/audio/s.flac", {"Range": "bytes=90-"}).read() == bytes(range(90, 100))
    assert get("/audio/s.flac").status == 200
    with pytest.raises(urllib.error.HTTPError) as ei:
        get("/api/nothing")
    assert ei.value.code == 404


def test_scan_prunes_deleted_files_and_sweeps_stale_temp(folder):
    import os
    import time
    tmp, _, _ = folder
    S.lib().scan()
    (tmp / "b.wav").unlink()
    S.lib().scan()
    assert "b.wav" not in read_json(S.lib().state / "hashes.json")
    stale = S.lib().state / "renders" / "x.abc.raw.wav"
    stale.write_bytes(b"x")
    os.utime(stale, (time.time() - 7200,) * 2)
    Library(tmp)
    assert not stale.exists()


def test_player_version_stamp_changes_with_the_sidecar(folder):
    r = S.lib().state / "renders"
    (r / "s.json").write_text('{"ok": 1}')
    url = start(S.lib(), 18800)["url"].rstrip("/")
    a = json.load(urllib.request.urlopen(url + "/api/version"))["stamp"]
    (r / "s.json").write_text('{"ok": 22}')
    assert json.load(urllib.request.urlopen(url + "/api/version"))["stamp"] != a
