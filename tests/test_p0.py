import json

import numpy as np
import pytest
from mcp.server.fastmcp.exceptions import ToolError
from synth import make_track

from openmixing import validate
from openmixing.core.errors import OpenMixingError, classify, structured
from openmixing.core.units import SR, to_bars
from openmixing.library import Library
from openmixing.model import Clip, SetModel

E = {"bpm": 120.0, "first_downbeat": 0.5, "file": "t.wav"}
SECS = [{"label": "intro", "start_bar": 0}, {"label": "drop", "start_bar": 16}]


def test_units():
    f = lambda v, **k: to_bars(v, E, k.get("cues"), SECS)
    assert f(8) == 8 and f("8") == 8 and f("bar 8") == 8 and f("8 bars") == 8
    assert f("64 beats") == 16
    assert f("4.5s") == pytest.approx(4.0 / 2.0)  # 4 s after the downbeat at 120 bpm = 2 bars
    assert f("section:drop") == 16 and f("drop+4") == 20 and f("section:drop-2") == 14
    assert f("cue:hook", cues={"hook": 33}) == 33 and f("hook", cues={"hook": 33}) == 33
    with pytest.raises(OpenMixingError) as ei:
        f("nowhere")
    assert ei.value.code == "invalid_position" and ei.value.fix


def test_structured_errors():
    @structured
    def boom():
        raise KeyError("no track matches 'x'")
    with pytest.raises(ToolError) as ei:
        boom()
    err = json.loads(str(ei.value))["error"]
    assert err["code"] == "not_found" and "x" in err["message"] and err["fix"]
    assert classify(ValueError("bad")).code == "invalid_value"


def test_content_id_survives_rename_and_flags_duplicates(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=4)
    lib = Library(tmp_path)
    first = lib.scan()[0]["id"]
    (tmp_path / "a.wav").rename(tmp_path / "renamed.wav")
    assert lib.scan()[0]["id"] == first
    (tmp_path / "copy.wav").write_bytes((tmp_path / "renamed.wav").read_bytes())
    rows = lib.scan()
    assert len({r["id"] for r in rows}) == 2
    assert [r["duplicate_of"] for r in rows if "duplicate_of" in r] == ["copy.wav"]


def test_cues_roundtrip(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=8)
    lib = Library(tmp_path)
    tid = lib.scan()[0]["id"]
    lib.set_cue(tid, "hook", 4.0)
    assert lib.to_bars(tid, "cue:hook+1") == 5.0
    lib.delete_cue(tid, "hook")
    with pytest.raises(OpenMixingError) as ei:
        lib.delete_cue(tid, "hook")
    assert ei.value.code == "not_found"


def test_beat_drift_and_silence():
    info = lambda t: {"bpm": 120.0}
    m = SetModel("s", [Clip("a", "t", length_beats=32), Clip("b", "t", length_beats=32)])
    m.clips[0].start, m.clips[1].start = {"at_sec": 0.0}, {"at_sec": 4.0}
    assert validate.beat_drift(m, info)[0]["max_drift_ms"] < 1
    m.clips[1].start = {"at_sec": 4.0, "offset_ms": 40}
    d = validate.beat_drift(m, info)[0]
    assert d["max_drift_ms"] == pytest.approx(40, abs=1) and not d["ok"]

    x = np.zeros((2, SR * 3), np.float32)
    x[:, : SR] = 0.1
    rows = [{"clip": "a", "start_sec": 0.0, "end_sec": 3.0}]
    s = validate.silences(x, 0.0, rows)
    assert s[0]["planned"] is False
    assert validate.silences(x, 0.0, [{"clip": "a", "start_sec": 0.0, "end_sec": 1.0}])[0]["planned"] is True
