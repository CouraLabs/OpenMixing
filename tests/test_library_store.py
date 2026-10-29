import json

import pytest
from synth import make_track

from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.jsonio import write_json
from openmixing.library import Library
from openmixing.library.store import History, check, match
from openmixing.model import Clip, SetModel


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    make_track(tmp_path / "b.wav", bpm=128, bars=16, bass_hz=60.0)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    lib.analyze("b.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def r(f, op, v):
    return {"field": f, "op": op, "value": v}


F = {"bpm": 124.0, "camelot": "8A", "energy": 6, "rating": 4, "tag": ["peak", "vocal"], "artist": "Foo Bar", "last_played": 1000.0, "has_stems": True}


@pytest.mark.parametrize("rule,want", [
    (r("bpm", "between", [120, 125]), True), (r("bpm", "gte", 125), False), (r("bpm", "lte", 125), True), (r("energy", "eq", 6), True),
    (r("energy", "ne", 6), False), (r("camelot", "in", ["8a", "9A"]), True), (r("camelot", "compatible_with", {"camelot": "9A"}), True),
    (r("camelot", "compatible_with", {"camelot": "11A"}), False), (r("bpm", "compatible_with", {"bpm": 62, "percent": 1}), True),
    (r("bpm", "compatible_with", {"bpm": 140}), False), (r("tag", "contains", "peak"), True), (r("tag", "in", ["x", "vocal"]), True),
    (r("tag", "contains", "x"), False), (r("artist", "contains", "foo"), True), (r("has_stems", "eq", True), True),
    (r("last_played", "within_days", 1), True), (r("play_count", "gte", 1), False), (r("color", "eq", "red"), False),
])
def test_rule_truth_table(rule, want):
    assert match(rule, F, now=1000.0 + 3600) is want


def test_groups_nest():
    g = {"all": [r("bpm", "gte", 120), {"any": [r("tag", "contains", "x"), r("rating", "gte", 4)]}]}
    assert match(g, F) and not match({"all": [g, r("energy", "gte", 9)]}, F)


@pytest.mark.parametrize("rules,path", [
    ({"all": [r("bpm", "gte", 1), r("loudness", "gte", 1)]}, "rules.all[1].field"),
    ({"any": [{"all": [r("bpm", "contains", 1)]}]}, "rules.any[0].all[0].op"),
    ({"all": [r("bpm", "between", 3)]}, "rules.all[0].value"), ({"all": []}, "rules.all"), ({"x": 1}, "rules.x"),
])
def test_unknown_rule_gives_structured_error_with_path(rules, path):
    with pytest.raises(OpenMixingError) as e:
        check(rules)
    assert e.value.field == path and e.value.fix


def test_crate_uses_sidecar_and_history(lib):
    S.set_tags("a.wav", rating=5, tags=["Peak "], color="red")
    S.set_tags("b.wav", rating=2, artist="Zed")
    S.save_crate("fav", {"all": [r("rating", "gte", 4)]})
    assert [t["file"] for t in S.get_crate("fav")["tracks"]] == ["a.wav"]
    S.save_crate("fast", {"all": [r("bpm", "gte", 125)]}, sort="bpm")
    assert [t["file"] for t in S.get_crate("fast")["tracks"]] == ["b.wav"]
    S.save_crate("peak", {"all": [r("tag", "contains", "peak")]})
    assert S.get_crate("peak")["total"] == 1 and S.find_tracks(crate="fast")["count"] == 1
    assert S.find_tracks(tag="PEAK")["count"] == 1
    with pytest.raises(OpenMixingError) as e:
        S.save_crate("bad", {"all": [r("album", "eq", "x")]})
    assert e.value.field == "rules.all[0].field" and "bad" not in lib.store.crates()


def test_embedded_tags_are_not_queried(lib):
    S.save_crate("art", {"all": [r("artist", "eq", "Anyone")]})
    assert S.get_crate("art")["total"] == 0


def test_collections_survive_restart_and_file_rename(lib, tmp_path):
    S.create_collection("warm", "opening")
    S.collection_add("warm", ["a.wav", "b.wav", "a.wav"])
    S.collection_remove("warm", ["b.wav"])
    assert S.list_collections()["collections"] == [{"name": "warm", "count": 1, "notes": "opening"}]
    (tmp_path / "a.wav").rename(tmp_path / "renamed.wav")
    fresh = Library(tmp_path)
    S.STATE["lib"] = fresh
    row = S.list_collections("warm")["tracks"][0]
    assert row["file"] == "renamed.wav" and row["bpm"] == pytest.approx(120, abs=1)


def test_tag_track_adds_removes_normalised(lib):
    S.set_tags("a.wav", tags=["x"], rating=3)
    assert S.tag_track("a.wav", add=["  Deep", "x"], remove=["X"])["tags"] == ["deep"]
    assert lib.user_tags(lib.resolve("a.wav")["id"])["rating"] == 3


def test_record_play_feeds_history_and_crates(lib):
    assert S.get_history()["total"] == 0
    S.cur().clips.append(Clip("c1", lib.resolve("a.wav")["id"], length_beats=32.0))
    S.cur().clips.append(Clip("c2", lib.resolve("b.wav")["id"], length_beats=32.0, start={"at_sec": 30.0}))
    S.record_play()
    S.record_play(tag="rendered")
    h = S.get_history(track="a.wav")
    assert h["total"] == 1 and h["plays"][0]["file"] == "a.wav" and h["plays"][0]["set"] == "t"
    assert S.get_history(since_days=1)["total"] == 2 and S.get_history(since_days=0)["total"] in (0, 2)
    S.save_crate("fresh", {"all": [r("last_played", "within_days", 1)]})
    S.save_crate("never", {"all": [r("play_count", "eq", 0)]})
    assert S.get_crate("fresh")["total"] == 2 and S.get_crate("never")["total"] == 0
    assert Library(lib.root).store.history.stats()[lib.resolve("a.wav")["id"]]["play_count"] == 1


def test_no_history_without_record_play(lib):
    S.cur().clips.append(Clip("c1", lib.resolve("a.wav")["id"], length_beats=32.0))
    assert not (lib.state / "history.jsonl").exists()


def test_history_tolerates_truncated_line_and_appends(tmp_path):
    h = History(tmp_path / "h.jsonl")
    h.append("s", [{"id": "x", "start_sec": 0, "end_sec": 1}], ts=5.0)
    with open(h.path, "a") as f:
        f.write('{"ts": 9, "set": "s", "tra')
    h.append("s", [{"id": "x", "start_sec": 0, "end_sec": 1}], ts=7.0)
    assert h.stats() == {"x": {"play_count": 2, "last_played": 7.0}}


def test_atomic_write_keeps_original_on_crash(lib, monkeypatch):
    S.create_collection("keep")
    before = lib.store.cfile.read_text()
    monkeypatch.setattr(json, "dumps", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        write_json(lib.store.cfile, {"x": 1})
    assert lib.store.cfile.read_text() == before


def test_writes_stay_in_state_dir(lib, tmp_path):
    S.create_collection("c")
    S.save_crate("k", {"all": [r("bpm", "gte", 1)]})
    S.cur().clips.append(Clip("c1", lib.resolve("a.wav")["id"], length_beats=32.0))
    S.record_play()
    assert all(p.parent == lib.state for p in (lib.store.cfile, lib.store.rfile, lib.store.history.path))
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["a.wav", "b.wav"]


def test_capabilities_lists_library():
    c = S.capabilities()["library"]
    assert "bpm" in c["crates"]["fields"] and c["history"]["written_by"] == "record_play"
