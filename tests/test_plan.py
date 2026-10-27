import json

import pytest

from openmixing import plan as P
from openmixing.library import Library
from openmixing.setmodel import SetModel
from synth import make_track


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("plan")
    make_track(d / "a.wav", bpm=124, bars=32)
    make_track(d / "b.wav", bpm=124, bars=32, bass_hz=65)
    lib = Library(d)
    lib.analyze("a.wav")
    lib.analyze("b.wav")
    return lib


def good_plan():
    return {"version": "1.0", "name": "p", "clips": [
        {"id": "one", "track": "a.wav", "src_start": 0, "bars": 16, "bpm": 124,
         "eq": {"low": [{"at": 8, "gain_db": "kill", "ramp_bars": 2, "curve": "equal_power"}]},
         "filter": [{"at": 12, "value": -0.8, "ramp_bars": 4, "curve": "power", "exponent": 3}],
         "fx": [{"fx": "echo", "at": 12, "bars": 4, "beats": "1/2", "wet": 0.5}],
         "loops": [{"at": 14, "size_beats": 4, "repeats": 2, "roll": True}]},
        {"id": "two", "track": "b.wav", "src_start": "0 beats", "bars": 16,
         "start": {"over": "one", "at_bar": 8},
         "volume": [{"at": 0, "level": 0}, {"at": 8, "level": 1, "ramp_bars": 4, "curve": "log"}],
         "eq": {"high": {"points": [[0, -80], [4, 0]]}}}]}


def test_valid_plan_builds_and_round_trips(lib):
    m = P.apply_plan(good_plan(), lib)
    assert [c.id for c in m.clips] == ["one", "two"]
    assert m.clips[1].start == {"anchor": "one", "anchor_beat": 32.0}
    assert m.clips[0].lanes["eq_low"].value_at(10 * 4) == pytest.approx(-80)
    assert m.clips[0].loops[0].length_beats == 8 and m.clips[0].fx[0].beats == 0.5
    again = P.apply_plan(json.loads(json.dumps(P.export_plan(m))), lib)
    assert again.to_dict() == m.to_dict()
    assert SetModel.from_dict(again.to_dict()).to_dict() == m.to_dict()


def test_curves_shape_the_envelope(lib):
    m = P.apply_plan(good_plan(), lib)
    f = m.clips[0].lanes["filter"]
    mid = f.value_at(14 * 4)  # halfway through a u^3 ramp to -0.8 stays near 0
    assert -0.2 < mid < 0
    assert f.value_at(16 * 4) == pytest.approx(-0.8)


def test_all_errors_reported_with_paths(lib):
    bad = good_plan()
    bad["clips"][0]["eq"]["low"][0]["gain_db"] = -200
    bad["clips"][0]["fx"][0].update(fx="nope", beats="3")
    bad["clips"][0]["loops"].append({"at": 14.5, "size_beats": 4, "bars": 1})
    bad["clips"][0]["warp"] = 1
    bad["clips"][1]["start"] = {"over": "ghost", "at_bar": 2}
    bad["clips"][1]["filter"] = [{"at": 99, "value": 3, "curve": "wobble"}]
    bad["clips"].append({"track": "missing.wav"})
    errs = P.validate_plan(bad, lib)
    got = {(e["code"], e["field"]) for e in errs}
    for want in [("out_of_range", "clips[0].eq.low[0].gain_db"), ("unknown_fx", "clips[0].fx[0].fx"), ("bad_beats", "clips[0].fx[0].beats"),
                 ("loop_overlap", "clips[0].loops"), ("unknown_field", "clips[0].warp"), ("unknown_clip", "clips[1].start.over"),
                 ("out_of_range", "clips[1].filter[0].at"), ("out_of_range", "clips[1].filter[0].value"),
                 ("unknown_curve", "clips[1].filter[0].curve"), ("track_not_found", "clips[2].track")]:
        assert want in got, (want, sorted(got))
    assert all(e["message"] and "fix" in e for e in errs)


def test_apply_is_atomic_and_version_checked(lib):
    bad = good_plan()
    bad["clips"][1]["gain_db"] = 999
    with pytest.raises(P.PlanError) as ei:
        P.apply_plan(bad, lib)
    assert ei.value.errors[0]["field"] == "clips[1].gain_db"
    assert P.validate_plan({"version": "9", "clips": []}, lib)[0]["code"] == "unsupported_version"
    cyc = {"version": "1.0", "clips": [{"id": "x", "track": "a.wav", "bars": 8, "start": {"over": "y", "at_bar": 1}},
                                        {"id": "y", "track": "b.wav", "bars": 8, "start": {"over": "x", "at_bar": 1}}]}
    assert "anchor_cycle" in {e["code"] for e in P.validate_plan(cyc, lib)}


def test_server_tools_do_not_change_set_on_failure(lib):
    from openmixing import server as s
    s._state["lib"] = lib
    s._state["set"] = SetModel("keep")
    bad = good_plan()
    bad["clips"][0]["bpm"] = 1
    r = s.apply_plan(bad)
    assert r["applied"] is False and s.cur().name == "keep" and not s.cur().clips
    assert s.apply_plan(good_plan(), dry_run=True)["applied"] is False and not s.cur().clips
    assert s.apply_plan(good_plan())["applied"] is True and len(s.cur().clips) == 2
    assert s.get_plan_schema()["version"] == "1.0"
    assert s.validate_plan(s.export_plan())["ok"]
