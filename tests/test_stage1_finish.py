import json
import subprocess

import pytest
from synth import make_track

from openmixing import server as S
from openmixing.analysis.meta import format_report
from openmixing.core.errors import OpenMixingError
from openmixing.library import Library
from openmixing.model import SetModel

CODECS = {  # extension -> ffmpeg output args
    "wav": [], "aiff": [], "flac": [], "mp3": ["-c:a", "libmp3lame"], "m4a": ["-c:a", "aac"],
    "ogg": ["-c:a", "vorbis", "-strict", "-2"], "opus": ["-c:a", "libopus"],
}


def encode(src, dst, args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *args, str(dst)], check=True)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("fmt")
    make_track(d / "src.wav", bpm=120, bars=8)
    for ext, args in CODECS.items():
        if ext != "wav":
            encode(d / "src.wav", d / f"t.{ext}", args)
    (d / "t.wav").write_bytes((d / "src.wav").read_bytes())
    (d / "src.wav").unlink()
    return d


@pytest.fixture
def server(folder):
    S.STATE.update(lib=Library(folder), set=None)
    yield
    S.STATE.update(lib=None, set=None)


@pytest.mark.parametrize("ext", list(CODECS))
def test_format_report_every_format(folder, ext):
    r = format_report(str(folder / f"t.{ext}"))
    assert r["supported"] and r["codec"] and r["channels"] == 2 and r["sample_rate"] and float(r["duration_sec"]) > 10
    assert r["lossless"] == (ext in ("wav", "aiff", "flac"))
    assert r["tags_writable"] is True


def test_unsupported_and_corrupt(tmp_path):
    (tmp_path / "x.wma").write_bytes(b"x")
    with pytest.raises(OpenMixingError) as e:
        format_report(str(tmp_path / "x.wma"))
    assert e.value.code == "unsupported_format" and e.value.fix
    (tmp_path / "bad.mp3").write_bytes(b"not audio at all" * 100)
    with pytest.raises(OpenMixingError) as e:
        format_report(str(tmp_path / "bad.mp3"))
    assert e.value.code == "corrupt_file"


def test_check_files_lists_problems_without_failing(tmp_path):
    make_track(tmp_path / "ok.wav", bpm=120, bars=8)
    (tmp_path / "bad.mp3").write_bytes(b"junk" * 500)
    (tmp_path / "x.wma").write_bytes(b"x")
    S.STATE.update(lib=Library(tmp_path), set=None)
    try:
        r = S.check_files()
    finally:
        S.STATE.update(lib=None)
    rows = {f["file"]: f for f in r["files"]}
    assert r["unusable"] == 1 and rows["ok.wav"]["supported"] and rows["bad.mp3"]["code"] == "corrupt_file" and r["ignored_audio_files"] == ["x.wma"]


def test_analyze_rejects_corrupt_with_structured_error(tmp_path):
    (tmp_path / "bad.mp3").write_bytes(b"junk" * 500)
    with pytest.raises(OpenMixingError) as e:
        Library(tmp_path).analyze("bad.mp3")
    assert e.value.code == "corrupt_file"


def test_set_tags_sidecar_only_by_default(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=8)
    before = (tmp_path / "a.wav").read_bytes()
    S.STATE.update(lib=Library(tmp_path), set=None)
    try:
        r = S.set_tags("a.wav", title="Hi", artist="Me", rating=4, color="red", tags=["peak", "vocal"], isrc="US-RC1-76-07839")
        assert r["mutated_file"] is False and r["tags"]["isrc"] == "USRC17607839"
        assert (tmp_path / "a.wav").read_bytes() == before
        row = S.list_tracks()["tracks"][0]
        assert row["rating"] == 4 and row["color"] == "red" and row["tags"] == ["peak", "vocal"]
        S.lib().analyze("a.wav")
        row = S.list_tracks()["tracks"][0]
        assert row["title"] == "Hi" and row["artist"] == "Me"
        assert [t["file"] for t in S.find_tracks(rating_min=4, tag="peak")["tracks"]] == ["a.wav"]
        assert S.find_tracks(rating_min=5)["tracks"] == [] and S.find_tracks(color="blue")["tracks"] == []
        S.set_tags("a.wav", clear=["rating"])
        assert "rating" not in S.lib().user_tags(S.lib().resolve("a.wav")["id"])
    finally:
        S.STATE.update(lib=None)


@pytest.mark.parametrize("bad", [{"rating": 9}, {"color": "teal"}, {"year": "abc"}, {"isrc": "nope"}, {"tags": "x"}])
def test_set_tags_validation(tmp_path, bad):
    make_track(tmp_path / "a.wav", bpm=120, bars=4)
    with pytest.raises(OpenMixingError) as e:
        Library(tmp_path).set_user_tags("a.wav", bad)
    assert e.value.code == "invalid_value" and e.value.field in bad


@pytest.mark.parametrize("ext", list(CODECS))
def test_write_to_file_every_format(folder, tmp_path, ext):
    d = tmp_path / ext
    d.mkdir()
    (d / f"t.{ext}").write_bytes((folder / f"t.{ext}").read_bytes())
    lib = Library(d)
    tid = lib.scan()[0]["id"]
    r = lib.set_user_tags("t." + ext, {"title": "T!", "artist": "A", "album": "Al", "genre": "House", "year": 2024, "comment": "c",
                                       "label": "Lbl", "isrc": "USRC17607839", "rating": 3}, write_to_file=True)
    assert r["mutated_file"] and r["backup"] and (d / ".openmixing" / "backups" / tid / f"t.{ext}").exists()
    assert r["skipped"]["rating"].startswith("sidecar")
    assert lib.scan()[0]["id"] == tid  # id pinned after the write
    from openmixing.analysis.meta import read_tags
    got = read_tags(str(d / f"t.{ext}"))
    assert got["title"] == "T!" and got["artist"] == "A" and got["genre"] == "House"


def test_backup_keeps_first_original_and_cache_survives(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=8)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    orig = (tmp_path / "a.wav").read_bytes()
    lib.set_user_tags("a.wav", {"title": "one"}, write_to_file=True)
    lib.set_user_tags("a.wav", {"title": "two"}, write_to_file=True)
    t = lib.scan()[0]
    assert (tmp_path / ".openmixing" / "backups" / t["id"] / "a.wav").read_bytes() == orig
    assert lib.cached(t) is not None and lib.analyze("a.wav")["tags"]["title"] == "two"  # no re-analysis needed
    r = lib.set_user_tags("a.wav", {"title": "three"}, write_to_file=True, backup=False)
    assert r["backup"] is None


def test_tracklist_text_json_cue(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=8)
    make_track(tmp_path / "b.wav", bpm=120, bars=8)
    lib = Library(tmp_path)
    lib.set_user_tags("a.wav", {"title": "Alpha", "artist": 'Ann "A"', "isrc": "USRC17607839"})
    S.STATE.update(lib=lib, set=SetModel("mix"))
    try:
        S.add_clip("a.wav", bars=8)
        S.add_clip("b.wav", bars=8)
        t = S.export_tracklist("text")
        lines = t["content"].splitlines()
        assert lines[2].startswith("00:00  Ann \"A\" - Alpha") and lines[3].startswith("00:16  b")  # 8 bars at 120 bpm = 16 s
        j = S.export_tracklist("json")
        assert [r["time_sec"] for r in j["content"]] == [0.0, pytest.approx(16.0, abs=0.05)] and json.loads(open(j["path"]).read())["audio_file"] == "mix.mp3"
        c = S.export_tracklist("cue")["content"]
        assert 'TRACK 02 AUDIO' in c and "INDEX 01 00:16:0" in c and "ISRC USRC17607839" in c and 'FILE "mix.mp3" MP3' in c
        assert (tmp_path / ".openmixing" / "renders" / "mix.cue").exists()
        with pytest.raises(KeyError):
            S.export_tracklist("text", set_name="nope")
    finally:
        S.STATE.update(lib=None, set=None)


def test_capabilities_lists_formats(folder, server):
    c = S.capabilities()
    assert c["formats"]["opus"]["tags_writable"] and c["formats"]["aac"]["tags_writable"] is False
    assert "rating" in c["tag_fields"]["sidecar_only"]


def test_apply_transition_and_plan_accept_beats(tmp_path):
    make_track(tmp_path / "a.wav", bpm=124, bars=32)
    make_track(tmp_path / "b.wav", bpm=124, bars=32, bass_hz=65)
    lib = Library(tmp_path)
    S.STATE.update(lib=lib, set=SetModel("t"))
    try:
        a, b = S.add_clip("a.wav", bars=24), S.add_clip("b.wav", bars=24, start_at_sec=0)
        r = S.apply_transition(a["clip"], b["clip"], "filter_fade", bars="32 beats", at_bar="16 beats")
        assert r["bars"] == 8 and r["incoming_enters_at_bar_of_outgoing"] == 4
        from openmixing import plan as P
        p = {"version": "1.0", "name": "x", "clips": [{"id": "o", "track": "a.wav", "bars": 16, "volume": [{"at": 0, "level": 1, "ramp_bars": "8 beats"}],
                                                      "fx": [{"fx": "echo", "at": 2, "bars": "16 beats"}], "loops": [{"at": 4, "size_beats": 4, "bars": "8 beats"}]}]}
        assert P.validate_plan(p, lib) == [] or all(e["code"] != "wrong_type" for e in P.validate_plan(p, lib))
    finally:
        S.STATE.update(lib=None, set=None)
