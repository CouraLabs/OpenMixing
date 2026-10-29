import json

import numpy as np
import pytest
from synth import make_track

from openmixing import plan as P
from openmixing import server as S
from openmixing.core.units import SR
from openmixing.engine import render_clip, segments
from openmixing.library import Library
from openmixing.model import Clip, LoopEvent, RateEvent, SetModel, TempoMap
from openmixing.model.rate import advance, preset

BPM = 120.0
BEAT = 60 / BPM
INFO = {"bpm": BPM, "first_downbeat": 0.0, "gain_db": 0.0}


def render(clip: Clip, audio: np.ndarray) -> np.ndarray:
    return render_clip(clip, INFO, audio, TempoMap(clip, BPM))[0]


def tone(sec: float, hz: float = 440.0) -> np.ndarray:
    y = (0.3 * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)
    return np.stack([y, y])


def sweep(sec: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    y = (0.3 * np.sin(2 * np.pi * (200 * t + 150 * t * t))).astype(np.float32)
    return np.stack([y, y])


def clicks(sec: float) -> np.ndarray:
    y = np.zeros(int(sec * SR), np.float32)
    y[:: int(4 * BEAT * SR)] = 1.0
    return np.stack([y, y])


def click_times(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(x > 0.5) / SR


def ev(kind="custom", beat=4.0, dur=2.0, slip=False, curve=None):
    return RateEvent("r", beat, dur, curve or preset(kind), slip, kind)


def test_spinback_instantaneous_frequency_follows_speed():
    clip = Clip("c", "t", 0.0, 12.0, rate=[ev("spinback", 4.0, 2.0)])
    out = render(clip, tone(10.0))
    n0 = int(4 * BEAT * SR)
    for u in (0.15, 0.3, 0.45):
        c = n0 + int(u * 2 * BEAT * SR)
        w = out[c - 240: c + 240]
        zc = np.flatnonzero(np.diff(np.signbit(w).astype(int)) == 1)
        f = SR / np.diff(zc).mean()
        assert f == pytest.approx(440 * (1 - u ** 2.5), rel=0.03)


def test_reverse_plays_the_source_backwards():
    src = sweep(10.0)
    clip = Clip("c", "t", 4.0, 12.0, rate=[ev("reverse", 2.0, 2.0)])
    out = render(clip, src)
    a, b = int(2 * BEAT * SR), int(4 * BEAT * SR)
    got = out[a + 2000: b - 2000]
    start = int(6 * BEAT * SR)
    ref = src[0][start - (b - a) + 2000 : start - 2000][::-1]
    n = min(len(got), len(ref))
    x = np.correlate(got[:n], ref[:n], "valid")[0] / (np.linalg.norm(got[:n]) * np.linalg.norm(ref[:n]))
    assert x > 0.99


def test_advance_and_segments_resume_rules():
    assert advance(ev("brake", dur=2.0)) == pytest.approx(1.0)
    assert advance(ev("reverse", dur=1.0)) == pytest.approx(-1.0)
    plain = segments(Clip("c", "t", 0.0, 16.0, rate=[ev("brake", 4.0, 2.0)]))
    assert plain[1][:4] == ("rate", 4.0, 6.0, 4.0) and plain[2][3] == pytest.approx(5.0)
    slip = segments(Clip("c", "t", 0.0, 16.0, rate=[ev("brake", 4.0, 2.0, True)]))
    assert slip[2][3] == pytest.approx(6.0)


def test_slip_recovers_phase_and_plain_is_offset_by_the_integral():
    src = clicks(12.0)
    after = lambda x: click_times(x[int(8 * BEAT * SR):]) + 8 * BEAT
    slipped = after(render(Clip("c", "t", 0.0, 14.0, rate=[ev("brake", 4.0, 2.0, True)]), src))
    assert slipped[0] == pytest.approx(8 * BEAT, abs=0.002)
    plain = after(render(Clip("c", "t", 0.0, 14.0, rate=[ev("brake", 4.0, 2.0)]), src))
    assert plain[0] == pytest.approx(9 * BEAT, abs=0.002)


def test_event_boundaries_are_clean():
    clip = Clip("c", "t", 0.0, 12.0, rate=[ev("spinback", 4.0, 2.0)])
    out = render(clip, tone(10.0))
    assert np.isfinite(out).all()
    for beat in (4.0, 6.0):
        c = int(beat * BEAT * SR)
        assert np.abs(np.diff(out[c - 50: c + 50])).max() < 0.1


def test_rate_events_may_not_overlap_each_other_or_loops():
    c = Clip("c", "t", 0.0, 16.0, rate=[ev(beat=4.0, dur=2.0, curve=preset("brake")), RateEvent("s", 5.0, 2.0, preset("brake"))])
    with pytest.raises(ValueError, match="r"):
        c.validate_rate()
    c = Clip("c", "t", 0.0, 16.0, rate=[ev("brake", 4.0, 2.0)], loops=[LoopEvent("l", 5.0, 1.0, 2.0)])
    with pytest.raises(ValueError):
        segments(c)


def test_clip_round_trip_with_rate():
    c = Clip("c", "t", 0.0, 16.0, rate=[ev("spinback", 4.0, 2.0, True)])
    assert Clip.from_dict(c.to_dict()).rate == c.rate
    d = c.to_dict()
    del d["rate"]
    assert Clip.from_dict(d).rate == []


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("rate")
    make_track(d / "a.wav", bpm=124, bars=32)
    lib = Library(d)
    lib.analyze("a.wav")
    return lib


def rate_plan(version="1.1"):
    return {"version": version, "clips": [
        {"id": "one", "track": "a.wav", "src_start": 0, "bars": 16, "bpm": 124,
         "rate": [{"id": "sb", "kind": "spinback", "at": 6, "bars": 1},
                  {"id": "rv", "kind": "reverse", "at": 8, "bars": "2 beats", "slip": True},
                  {"id": "cu", "at": "12 beats", "bars": 1, "points": [[0, 1], [0.5, -2], [1, 0.5]]}]}]}


def test_plan_round_trip_and_versions(lib):
    m = P.apply_plan(rate_plan(), lib)
    r = m.clips[0].rate
    assert [e.kind for e in r] == ["spinback", "reverse", "custom"] and r[1].slip and r[1].duration_beats == 2.0
    out = json.loads(json.dumps(P.export_plan(m)))
    assert out["version"] == "1.1" and out["clips"][0]["rate"][2]["points"] == [[0, 1], [0.5, -2], [1, 0.5]]
    assert P.apply_plan(out, lib).to_dict() == m.to_dict()
    plain = rate_plan("1.0")
    del plain["clips"][0]["rate"]
    assert P.export_plan(P.apply_plan(plain, lib))["version"] == "1.0"


def test_plan_errors_are_reported_with_paths(lib):
    bad = rate_plan()
    bad["clips"][0]["rate"] = [{"kind": "wobble", "at": 1, "bars": 1}, {"at": 2, "bars": 1, "points": [[0, 9], [1, 0]]},
                               {"at": 4, "bars": 2, "kind": "brake"}, {"at": 5, "bars": 2, "kind": "brake"}]
    fields = {e["field"] for e in P.validate_plan(bad, lib)}
    assert {"clips[0].rate[0].kind", "clips[0].rate[1].points", "clips[0].rate"} <= fields


def test_tools_and_audit(lib):
    S.STATE.update(lib=lib, set=SetModel("t"))
    try:
        cid = S.add_clip("a.wav", bars=16, start_at_sec=0)["clip"]
        r = S.add_rate(cid, 6, 1, "spinback")
        assert r["ends_at_bar"] == 7.0
        with pytest.raises(ValueError):
            S.add_rate(cid, 6.5, 1, "brake")
        with pytest.raises(ValueError):
            S.add_rate(cid, 10, 1, "custom")
        assert any("spinback" in line for e in S.get_set(detail=True)["clips"] for line in e.get("events", []))
        assert not [i for i in S.audit_set()["issues"] if i["severity"] == "error"]
        S.add_loop(cid, 12, 4, repeats=1)
        S.add_rate(cid, 13, 1, "brake")
        assert any("rate event" in i["message"] for i in S.audit_set()["issues"])
        assert S.remove_rate(cid, r["rate_id"])["remaining_rate"] != []
        S.clear_automation(cid)
        assert not S.clip_of(cid).rate
    finally:
        S.STATE.update(lib=None, set=None)
