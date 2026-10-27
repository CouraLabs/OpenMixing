import pytest

from openmixing import plan as P, server as S
from openmixing.errors import OpenMixingError
from openmixing.library import Library
from openmixing.setmodel import SetModel
from openmixing.units import duration_bars, is_output_unit, out_bars
from synth import make_track


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=32)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    tid = lib.resolve("a.wav")["id"]
    lib.set_cue(tid, "hook", 12.0)
    S._state.update(lib=lib, set=SetModel("t"))
    yield lib
    S._state.update(lib=None, set=None)


def test_output_vs_source_units():
    assert is_output_unit(8) and is_output_unit("8") and is_output_unit("8 bars") and is_output_unit("32 beats")
    assert not is_output_unit("cue:hook") and not is_output_unit("95.5s") and not is_output_unit("section:drop+4")
    assert out_bars("32 beats", 4.0, None) == 8.0                       # output units ignore src_start
    assert out_bars("cue:x", 4.0, lambda v: 12.0) == 8.0                 # source units subtract it
    with pytest.raises(OpenMixingError) as e:
        out_bars("cue:x", 10.0, lambda v: 4.0)
    assert e.value.code == "out_of_range"
    with pytest.raises(OpenMixingError):
        out_bars("cue:x", 0.0, None)
    assert duration_bars("16 beats") == 4.0 and duration_bars(2) == 2.0
    with pytest.raises(OpenMixingError):
        duration_bars("cue:x")


def test_tools_accept_flexible_positions(lib):
    c = S.add_clip("a.wav", src_start_bar=4, bars=24)["clip"]
    S.set_eq(c, "low", "cue:hook", "kill", ramp_bars="8 beats")          # source bar 12 -> output bar 8, ramp 2 bars
    clip = S.cur().clip(c)
    assert clip.lanes["eq_low"].value_at(8 * 4 + 2 * 4) == pytest.approx(-80)
    S.set_filter(c, "16 beats", -0.5)                                    # output bar 4
    assert clip.lanes["filter"].value_at(4 * 4 + 1) == pytest.approx(-0.5)
    S.set_volume(c, 6, 0.5)                                              # numbers keep meaning output bars
    assert clip.lanes["volume"].value_at(6 * 4 + 1) == pytest.approx(0.5)
    fx = S.add_fx(c, "echo", "cue:hook", "4 beats", beats="1/2")
    e = clip.fx[-1]
    assert e.beat == 8 * 4 and e.duration_beats == 4
    lp = S.add_loop(c, "cue:hook+2", 4, bars="8 beats")
    assert clip.loops[-1].beat == 10 * 4 and lp["ends_at_bar"] == 12
    S.clear_automation(c, "eq_low", from_bar="cue:hook", to_bar="cue:hook+4")
    with pytest.raises(OpenMixingError):
        S.set_filter(c, "cue:nowhere", 0.0)
    with pytest.raises(OpenMixingError) as ei:
        S.set_filter(c, "cue:hook-20", 0.0)                              # before the clip starts playing
    assert ei.value.code == "out_of_range"


def test_anchor_uses_source_units_of_the_anchor_clip(lib):
    o = S.add_clip("a.wav", src_start_bar=4, bars=24)["clip"]
    n = S.add_clip("a.wav", bars=8, over_clip=o, at_bar="cue:hook")["clip"]
    assert S.cur().clip(n).start == {"anchor": o, "anchor_beat": 8 * 4}


def test_plan_accepts_source_positions(lib):
    plan = {"version": "1.0", "clips": [
        {"id": "x", "track": "a.wav", "src_start": 4, "bars": 24,
         "eq": {"low": [{"at": "cue:hook", "gain_db": "kill"}]},
         "fx": [{"fx": "echo", "at": "8 beats", "bars": 2}],
         "loops": [{"at": "cue:hook+2", "size_beats": 4, "repeats": 2}]},
        {"id": "y", "track": "a.wav", "bars": 8, "start": {"over": "x", "at_bar": "cue:hook"}}]}
    m = P.apply_plan(plan, lib)
    assert m.clips[0].lanes["eq_low"].value_at(8 * 4 + 1) == pytest.approx(-80)
    assert m.clips[0].loops[0].beat == 10 * 4
    assert m.clips[1].start["anchor_beat"] == 8 * 4
    bad = dict(plan, clips=[dict(plan["clips"][0], eq={"low": [{"at": "cue:nope", "gain_db": 0}]})])
    errs = P.validate_plan(bad, lib)
    assert errs and errs[0]["field"].endswith("eq.low[0].at")
    early = dict(plan, clips=[dict(plan["clips"][0], src_start=20)])
    assert any(e["code"] == "out_of_range" for e in P.validate_plan(early, lib))
