import asyncio

import pytest
from synth import make_track

from openmixing import mixing
from openmixing import plan as P
from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.units import BAR
from openmixing.library import Library
from openmixing.model import Clip, SetModel

R = 8.0


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=24)
    make_track(tmp_path / "b.wav", bpm=120, bars=24)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def clip(bars=12.0):
    return Clip("c", "a", length_beats=bars * BAR)


def build(c=None, **kw):
    c = c or clip()
    r = mixing.build_and_release(c, R, "t1", **kw)
    return c, r


def test_expansion_is_plain_primitives_tagged_with_a_group():
    c, r = build(build_bars=8, gap_beats=0.5)
    assert r["group"] == "tension:t1" and r["build_starts_at_bar"] == 0
    ev = [*c.fx, *c.loops, *c.triggers]
    assert ev and all(e.group == "tension:t1" for e in ev)
    kinds = {t.kind for t in c.triggers}
    assert kinds == {"riser", "snare", "impact"} and [l.size_beats for l in c.loops] == [1.0, 0.5, 0.25, 0.125] and all(l.roll for l in c.loops)
    riser = next(t for t in c.triggers if t.kind == "riser")
    assert riser.align == "end" and riser.beat == pytest.approx(R * BAR - 0.5) and riser.start == pytest.approx(0)
    assert max(t.start + t.duration_beats for t in c.triggers if t.kind == "snare") <= R * BAR - 0.5 + 1e-9
    assert next(t for t in c.triggers if t.kind == "impact").beat == R * BAR
    assert c.fx[0].fx == "reverb" and c.fx[0].beat + c.fx[0].duration_beats == pytest.approx(R * BAR - 0.5)


def test_lanes_ramp_gap_ends_on_the_downbeat_and_everything_resets():
    c, _ = build(build_bars=8, gap_beats=0.5)
    f, b, v = (c.lanes[k] for k in ("filter", "eq_low", "volume"))
    assert f.value_at(0) == pytest.approx(0) and 0.5 < f.value_at(R * BAR - 1) <= 0.85 and b.value_at(R * BAR - 1) < -25
    assert f.value_at(R * BAR / 2) < f.value_at(R * BAR - 1)
    assert v.value_at(R * BAR - 0.4) == 0 and v.value_at(R * BAR - 0.6) > 0.9
    assert v.value_at(R * BAR) == 0 and v.value_at(R * BAR + 0.05) == pytest.approx(1.0)
    for lane in (f, b):
        assert lane.value_at(R * BAR + 0.05) == pytest.approx(0)


def test_tension_levels_map_to_fixed_element_intensities():
    lo, _ = build(tension=[[0, 0.0], [R * BAR, 0.3]], build_bars=8)
    hi, _ = build(tension=[[0, 0.0], [R * BAR, 1.0]], build_bars=8)
    assert hi.lanes["filter"].value_at(R * BAR - 0.5) > lo.lanes["filter"].value_at(R * BAR - 0.5)
    assert mixing.techniques.mapped("filter", 1.0) == 0.85
    assert not lo.loops and len(hi.loops) == 4
    g = lambda c: next(t.gain_db for t in c.triggers if t.kind == "riser")
    assert g(hi) > g(lo)


def test_elements_and_errors():
    c, r = build(elements=["hp_eq"], impact=False, reverb_throw=False, gap_beats=0)
    assert not (c.fx or c.loops or c.triggers) and c.lanes["eq_low"].points and not c.lanes["filter"].points
    assert c.lanes["volume"].is_neutral()
    with pytest.raises(OpenMixingError) as e:
        build(elements=["laser"])
    assert e.value.code == "unknown_technique"
    for kw in ({"build_bars": 12}, {"gap_beats": 0.01}, {"gap_beats": 31.5}):
        with pytest.raises(OpenMixingError) as e:
            build(**kw)
        assert e.value.code == "out_of_range"


def test_existing_gap_is_detected_from_the_lead_in():
    secs = [{"start_bar": 16, "label": "drop", "lead_in": {"gap_beats": 1.0}}, {"start_bar": 32, "label": "drop", "lead_in": {"gap_beats": 0.1}},
            {"start_bar": 40, "label": "drop"}]
    assert mixing.existing_gap(secs, 16) == 1.0 and mixing.existing_gap(secs, 32) == 0 and mixing.existing_gap(secs, 40) == 0
    assert mixing.existing_gap(secs, 20) == 0


def test_plan_techniques_expand_round_trip_and_errors(lib):
    t = {"technique": "build_and_release", "id": "dropbuild", "resolve_at": 8, "build_bars": 4, "elements": ["riser", "hp_eq", "loop_roll"],
         "gap_beats": 0.5}
    plan = {"version": "1.1", "name": "w", "clips": [{"id": "x", "track": "a.wav", "bars": 12, "techniques": [t]}]}
    m = P.apply_plan(plan, lib)
    c = m.clips[0]
    assert {e.group for e in (*c.triggers, *c.loops)} >= {"tension:dropbuild"} and c.lanes["eq_low"].points
    out = P.export_plan(m)
    assert "techniques" not in out["clips"][0] and out["version"] == "1.1"
    assert all(x["group"] == "tension:dropbuild" for x in out["clips"][0]["triggers"] + out["clips"][0]["loops"])
    assert P.export_plan(P.apply_plan(out, lib)) == out
    assert P.export_plan(P.apply_plan(plan, lib)) == out
    bad = lambda **kw: P.validate_plan({**plan, "clips": [{**plan["clips"][0], "techniques": [{**t, **kw}]}]}, lib)
    assert bad(technique="nope")[0]["code"] == "unknown_technique"
    assert bad(elements=["laser"])[0]["code"] == "unknown_technique"
    assert bad(build_bars=20)[0]["field"].endswith("resolve_at")
    assert bad(peak=2)[0]["code"] == "out_of_range" and bad(gap_beats=9)[0]["code"] == "out_of_range"
    assert bad(tension=[[0, 2], [4, 1]])[0]["code"] == "out_of_range"
    assert P.validate_plan({**plan, "clips": [{**plan["clips"][0], "techniques": [{"technique": "build_and_release"}]}]}, lib)[0]["code"] == "missing_field"


def test_tools_build_and_remove_technique(lib):
    cid = S.add_clip("a.wav", bars=12)["clip"]
    r = S.build_tension(cid, 8, build_bars=4)
    assert r["group"].startswith("tension:") and r["gap_beats"] == 0.5 and "technique" in " ".join(S.get_set()["clips"][0]["events"])
    assert S.list_techniques()["elements"] and "build_and_release" in S.capabilities()["techniques"]["techniques"]
    out = S.remove_technique(cid, r["group"])
    c = S.cur().clips[0]
    assert out["removed"] >= 4 and not (c.fx or c.loops or c.triggers)
    assert all(c.lanes[k].is_neutral() for k in ("filter", "eq_low", "volume"))
    with pytest.raises(OpenMixingError):
        S.remove_technique(cid, r["group"])


def test_gap_beats_on_drop_swap_and_echo_out(lib):
    for style in ("drop_swap", "echo_out"):
        S.new_set(style)
        c1 = S.add_clip("a.wav", bars=16)["clip"]
        c2 = S.add_clip("b.wav", bars=16, over_clip=c1, at_bar=8)["clip"]
        S.apply_transition(c1, c2, style, bars=8, at_bar=8, gap_beats=1.0)
        v = S.cur().clips[0].lanes["volume"]
        assert v.value_at(16 * BAR - 0.5) == 0 and v.value_at(16 * BAR - 1.5) > 0.9
        S.apply_transition(c1, c2, style, bars=8, at_bar=8)
        assert S.cur().clips[0].lanes["volume"].value_at(16 * BAR - 0.5) > 0.9
    with pytest.raises(ValueError):
        S.apply_transition(c1, c2, "bass_swap", bars=8, at_bar=8, gap_beats=1.0)


def test_rendered_build_release_matches_target_and_has_the_gap(lib):
    plan = {"version": "1.1", "name": "br", "intent": {"pre_drop_gap_min_beats": 0.25, "tension_target": {
        "shape": "build_release", "min_correlation": 0.8, "start_sec": 0, "end_sec": 20}},
        "clips": [{"id": "x", "track": "b.wav", "bars": 11, "techniques": [{"technique": "build_and_release", "resolve_at": 8, "build_bars": 8}]}]}
    assert S.apply_plan(plan)["applied"]
    v = asyncio.run(S.render(format="wav"))["validation"]
    tm = v["checks"]["tension_match"]
    assert tm["correlation"] >= 0.8, tm
    gaps = v["checks"]["pre_drop_gap"]
    assert gaps and 0.25 <= gaps[0]["beats"] <= 0.6 and abs(gaps[0]["end_sec"] - 16.0) < 0.15, gaps
    assert {i["item"]: i["status"] for i in v["intent_report"]} == {"pre_drop_gap_min_beats": "pass", "tension_target": "pass"}
