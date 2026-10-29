import numpy as np
import pytest
from synth import make_track

from openmixing import candidates as C
from openmixing import plan as P
from openmixing import server as S
from openmixing.analysis.key import camelot
from openmixing.core.units import SR
from openmixing.dsp import stretch
from openmixing.engine import render_clip
from openmixing.library import Library
from openmixing.mixing import plan_key_match, shift_camelot
from openmixing.model import Clip, Lane, SetModel, TempoMap, shape, write

INFO = {"bpm": 120.0, "first_downbeat": 0.0, "gain_db": 0.0}


def peak_hz(x: np.ndarray) -> float:
    w = x * np.hanning(len(x))
    n = 8 * len(x)
    return float(np.fft.rfftfreq(n, 1 / SR)[np.argmax(np.abs(np.fft.rfft(w, n)))])


def tone(sec: float, hz: float = 440.0) -> np.ndarray:
    y = (0.3 * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)
    return np.stack([y, y])


@pytest.mark.parametrize("curve", ["equal_power", "log", "s_curve", "power"])
def test_curve_endpoints_and_monotone(curve):
    lane = Lane(120.0, [[0.0, 120.0]])
    write(lane, 4.0, 130.0, 8.0, curve, 3.0)
    ys = [p[1] for p in lane.points]
    assert lane.value_at(4.0) == pytest.approx(120.0) and lane.value_at(12.0) == pytest.approx(130.0)
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
    assert len(lane.points) > 8


def test_curve_shapes_differ_and_fall_mirrors():
    assert shape("s_curve", 2, True)(0.25) < 0.25 and shape("power", 3, True)(0.5) == pytest.approx(0.125)
    assert shape("log", 2, True)(0.5) > 0.5 and shape("log", 2, False)(0.5) < 0.5
    lane = Lane(120.0, [[0.0, 130.0]])
    write(lane, 0.0, 120.0, 8.0, "sharp_cut")
    assert lane.value_at(7.9) == pytest.approx(130.0, abs=0.2) and lane.value_at(8.1) == pytest.approx(120.0)


def test_s_curve_tempo_ramp_renders_on_the_tempo_map_grid():
    bpm = 120.0
    beat = 60.0 / bpm
    n = int(40 * beat * SR)
    y = np.zeros(n, np.float32)
    for k in range(40):
        i = int(k * beat * SR)
        m = int(0.03 * SR)
        y[i : i + m] = np.sin(2 * np.pi * 1000 * np.arange(m) / SR) * np.exp(-np.arange(m) / (0.006 * SR))
    audio = np.stack([y, y])
    clip = Clip("c", "t", 0.0, 32.0)
    clip.tempo.points = [[0.0, 120.0]]
    write(clip.tempo, 4.0, 132.0, 16.0, "s_curve")
    tm = TempoMap(clip, bpm)
    out = render_clip(clip, INFO, audio, tm)[0]
    env = np.abs(out)
    errs = []
    for k in range(2, 28):
        want = int(tm.beat_to_sec(k) * SR)
        lo = want - int(0.04 * SR)
        got = lo + int(np.argmax(env[lo : want + int(0.04 * SR)] > 0.25 * env[lo : want + int(0.04 * SR)].max()))
        errs.append(abs(got - want) / SR)
    assert max(errs) < 0.005


def test_pitch_glide_tracked_by_spectral_peak():
    sec = 6.0
    semi = np.concatenate([np.zeros(int(2 * SR)), np.linspace(0, 2, int(2 * SR)), np.full(int(2 * SR), 2.0)])
    y = stretch.play(tone(sec + 1), 0, 1.0, len(semi), semi)[0]
    f0, f1 = peak_hz(y[int(0.5 * SR) : int(1.5 * SR)]), peak_hz(y[int(4.5 * SR) : int(5.5 * SR)])
    mid = peak_hz(y[int(2.9 * SR) : int(3.1 * SR)])
    assert f0 == pytest.approx(440, abs=3) and f1 == pytest.approx(440 * 2 ** (2 / 12), abs=4)
    assert f0 + 10 < mid < f1 - 10


def test_pitch_chunks_are_capped_to_quarter_semitone_steps():
    semi = np.linspace(0, 2, 96_000)
    assert 11_000 <= stretch.pitch_chunk(semi, 48_000) <= 12_100
    assert stretch.pitch_chunk(np.linspace(0, 12, 96_000), 48_000) == 4096
    assert stretch.pitch_chunk(np.zeros(48_000), 48_000) == 48_000


def test_pitch_lane_in_clip_render_and_round_trip():
    clip = Clip("c", "t", 0.0, 16.0)
    write(clip.pitch, 8.0, 12.0, 0.0)
    clip.semitones = 0.0
    out = render_clip(clip, INFO, tone(9), TempoMap(clip, 120.0))[0]
    assert peak_hz(out[int(0.5 * SR) : int(1.5 * SR)]) == pytest.approx(440, abs=3)
    assert peak_hz(out[int(6.2 * SR) : int(7.5 * SR)]) == pytest.approx(880, abs=6)
    assert Clip.from_dict(clip.to_dict()).pitch.points == clip.pitch.points


@pytest.mark.parametrize("minor", [False, True])
def test_camelot_shift_table_matches_pitch_class_arithmetic(minor):
    for pc in range(12):
        for s in (-1, 0, 1):
            assert shift_camelot(camelot(pc, minor), s) == camelot((pc + s) % 12, minor)


def test_plan_key_match_variants():
    a = {"camelot": "8B", "key_conf": 0.9, "tuning_cents": 0.0}
    r = plan_key_match(a, {"camelot": "10B", "key_conf": 0.9})
    assert (r["semitones_in"], r["semitones_out"], r["camelot_after"], r["distance_after"]) == (-2.0, 0.0, "8B", 0)
    r = plan_key_match(a, {"camelot": "10B"}, prefer="outgoing")
    assert (r["semitones_in"], r["semitones_out"]) == (0.0, 2.0)
    r = plan_key_match(a, {"camelot": "10B"}, prefer="split")
    assert (r["semitones_in"], r["semitones_out"]) == (-1.0, 1.0)
    assert plan_key_match(a, {"camelot": "8A"})["semitones_in"] == 0 and plan_key_match(a, {"camelot": "9B"})["semitones_in"] == 0
    assert plan_key_match(a, {"camelot": "9B"}, mode="exact", max_shift=5)["semitones_in"] == 5.0
    assert plan_key_match(a, {"camelot": "3B"}, max_shift=3)["semitones_in"] != 0


def test_plan_key_match_tuning_confidence_and_quality():
    a = {"camelot": "8B", "key_conf": 0.9, "tuning_cents": 0.0}
    r = plan_key_match(a, {"camelot": "8B", "key_conf": 0.9, "tuning_cents": -30.0})
    assert r["cents_error"] == -30.0 and r["semitones_in"] == pytest.approx(0.3) and r["confidence"] < plan_key_match(a, a)["confidence"]
    assert plan_key_match(a, {"camelot": "10B", "key_conf": 0.2})["low_confidence"]
    assert "quality_note" in plan_key_match(a, {"camelot": "9B"}, mode="exact", max_shift=5)
    with pytest.raises(ValueError):
        plan_key_match(a, a, mode="nope")


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=32)
    make_track(tmp_path / "b.wav", bpm=120, bars=32)
    lib = Library(tmp_path)
    for f, cam in (("a.wav", "8B"), ("b.wav", "10B")):
        lib.analyze(f)
        lib.set_override(lib.resolve(f)["id"], camelot=cam)
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def test_tools_tempo_curve_pitch_and_match_key(lib):
    ia = S.add_clip("a.wav", bars=16, start_at_sec=0)["clip"]
    ib = S.add_clip("b.wav", bars=16, over_clip=ia, at_bar=8)["clip"]
    c = S.clip_of(ia)
    S.set_tempo(ia, 4, 126, ramp_bars=4, curve="s_curve")
    assert len(c.tempo.points) > 6 and c.tempo.value_at(32.0) == pytest.approx(126)
    S.sync_clip(ib, ia, at_bar=0, ramp_bars=2, ramp_curve="log")
    S.set_pitch(ia, 8, 1.5, ramp_bars=2, curve="s_curve")
    assert c.pitch.value_at(40.0) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        S.set_pitch(ia, 0, 13)
    r = S.match_key(ib, ia)
    assert r["applied"] and S.clip_of(ib).semitones == -2.0 and S.clip_of(ia).semitones == 0.0
    r = S.match_key(ib, ia, over_bars=4)
    assert S.clip_of(ib).semitones == 0.0 and S.clip_of(ib).pitch.value_at(16.0) == pytest.approx(-2.0)
    S.clear_automation(ia, "pitch")
    assert not c.pitch.points


def test_plan_key_match_and_pitch_round_trip(lib):
    plan = {"version": "1.0", "name": "k", "clips": [
        {"id": "x", "track": "a.wav", "bars": 8},
        {"id": "y", "track": "b.wav", "bars": 8, "key_match": {"to": "x"}, "pitch": [{"at": 4, "semitones": 1, "ramp_bars": 2, "curve": "s_curve"}]}]}
    m = P.apply_plan(plan, lib)
    y = m.clip("y")
    assert y.semitones == -2.0 and y.pitch.value_at(24.0) == pytest.approx(1.0)
    out = P.export_plan(m)["clips"][1]
    assert out["semitones"] == -2.0 and "key_match" not in out and out["pitch"]["points"]
    bad = {**plan, "clips": [plan["clips"][0], {**plan["clips"][1], "key_match": {"to": "zzz"}, "semitones": 1}]}
    assert {e["code"] for e in P.validate_plan(bad, lib)} >= {"unknown_clip", "conflicting_fields"}
    assert any(e["field"].endswith("pitch[0].semitones") for e in P.validate_plan(
        {**plan, "clips": [{**plan["clips"][0], "pitch": [{"at": 0, "semitones": 40}]}]}, lib))


def test_pair_report_exposes_match_and_candidates_carry_semitones(lib):
    rep = C.pair_report(C.track_facts(lib, "a.wav"), C.track_facts(lib, "b.wav"))
    assert rep["key"]["match"]["semitones_in"] == -2.0 and rep["key"]["semitones_to_match"] == -2
    r = C.suggest(lib, "a.wav", "b.wav", limit=3)
    shifted = [c for c in r["candidates"] if "key_match" in c]
    assert shifted and all(next(x for x in c["plan"]["clips"] if x["id"] == "in")["semitones"] == -2.0 for c in shifted)
