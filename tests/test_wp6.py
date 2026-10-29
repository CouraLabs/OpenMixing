import librosa
import numpy as np
import pytest
import soundfile as sf
from synth import make_track

from openmixing import plan as P
from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.units import SR
from openmixing.engine import render_clip, render_set
from openmixing.engine.sampler import load_samples
from openmixing.library import Library
from openmixing.model import SampleSlot, SetModel, TempoMap

INFO = {"bpm": 120.0, "first_downbeat": 0.0, "gain_db": 0.0}
SILENT = np.zeros((2, 40 * SR), np.float32)


def click(n=SR // 20):
    y = np.random.default_rng(1).standard_normal(n) * np.exp(-np.arange(n) / SR * 400)
    return y.astype(np.float32)


def wav(path, y, sr=SR):
    sf.write(path, np.stack([y, y]).T if y.ndim == 1 else y.T, sr)
    return path


def tone(sec, hz, amp=0.5):
    return (amp * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    wav(tmp_path / "click.wav", click())
    wav(tmp_path / "low.wav", tone(3.0, 200))
    wav(tmp_path / "high.wav", tone(1.0, 2000))
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def play(clip, bpm=120.0):
    model = S.cur()
    return render_clip(clip, {**INFO, "bpm": bpm}, SILENT, TempoMap(clip, bpm), None, (), load_samples(model))[:, : int(clip.length_beats * 60 / bpm * SR)]


def onsets(y):
    return librosa.onset.onset_detect(y=y[0], sr=SR, hop_length=32, units="time", backtrack=True)


def test_click_onsets_at_planned_beats_and_through_tempo_ramp(lib):
    S.load_sample("hit", "click.wav")
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    c = S.clip_of(a)
    for b in (4.0, 6.0, 9.0, 9.0, 13.0, 20.0, 26.0):
        S.trigger_sample(a, "hit", b / 4, 0.25)
    tm = TempoMap(c, 120.0)
    got, want = onsets(play(c)), np.array([tm.beat_to_sec(t.start) for t in c.triggers])
    assert np.abs(np.array([got[np.argmin(np.abs(got - w))] for w in want]) - want).max() < 0.005
    c.tempo.points = [[0.0, 120.0]]
    c.tempo.set(8.0, 120.0)
    c.tempo.set(16.0, 150.0, 8.0)
    tm = TempoMap(c, 120.0)
    got, want = onsets(play(c)), np.array([tm.beat_to_sec(t.start) for t in c.triggers])
    assert np.abs(np.array([got[np.argmin(np.abs(got - w))] for w in want]) - want).max() < 0.008


def test_stretch_to_beats_fills_exactly_and_loop_for_duration_repeats(lib):
    S.load_sample("lo", "low.wav")
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    S.trigger_sample(a, "lo", 0, 1, fit="stretch_to_beats", gain_db=0)
    y = play(S.clip_of(a))[0]
    last = int(np.nonzero(np.abs(y) > 0.01 * np.abs(y).max())[0][-1])
    assert abs(last - 2 * SR) < 0.001 * SR
    S.clip_of(a).triggers.clear()
    S.trigger_sample(a, "lo", 0, 2, fit="loop_for_duration", gain_db=0)
    y = play(S.clip_of(a))[0]
    assert np.abs(y[int(3.7 * SR) : int(3.9 * SR)]).max() > 0.1 and np.abs(y[int(4.05 * SR) : 5 * SR]).max() < 1e-6


def test_loop_follows_clip_tempo(lib):
    beat = 0.5
    y = np.zeros(int(4 * beat * SR), np.float32)
    for k in range(4):
        y[int(k * beat * SR) : int(k * beat * SR) + SR // 20] = click()
    wav(lib.root / "loop.wav", y)
    got = S.load_sample("lp", "loop.wav", "loop", root_bpm=120.0)
    assert got["root_bpm"] == 120.0 and got["seconds"] == pytest.approx(2.0)
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    S.trigger_sample(a, "lp", 0, 2, fit="loop_for_duration")
    on = onsets(play(S.clip_of(a), bpm=150.0))
    assert np.diff(on)[:6] == pytest.approx([0.4] * 6, abs=0.008)


def test_loop_bpm_detected_and_snapped(lib):
    beat = 60 / 128
    y = np.zeros(int(8 * beat * SR), np.float32)
    for k in range(8):
        y[int(k * beat * SR) : int(k * beat * SR) + SR // 20] = click() * (1.0 if k % 4 == 0 else 0.7)
    wav(lib.root / "l128.wav", y)
    assert S.load_sample("l", "l128.wav", "loop")["root_bpm"] == pytest.approx(128.0, rel=0.03)


def test_choke_cuts_earlier_voices_of_the_group(lib):
    S.load_sample("lo", "low.wav", choke_group="g")
    S.load_sample("hi", "high.wav", choke_group="g")
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    S.trigger_sample(a, "lo", 0, 2, gain_db=0)
    S.trigger_sample(a, "hi", 0.25, 1, gain_db=0, choke=True)
    c = S.clip_of(a)
    low = lambda y, a: np.abs(np.fft.rfft(y[0, a : a + 4800]))[20]
    y = play(c)
    assert low(y, 4800) > 100 and low(y, 28800 + 2400) < 1e-3 * low(y, 4800)
    c.triggers[1].params["choke"] = 0.0
    assert low(play(c), 28800 + 2400) > 0.5 * low(y, 4800)


def test_sample_paths_must_stay_in_tracks_folder_or_samples_dir(lib, tmp_path_factory):
    out = tmp_path_factory.mktemp("outside")
    wav(out / "x.wav", click())
    for bad in (str(out / "x.wav"), "../x.wav"):
        with pytest.raises(OpenMixingError) as e:
            S.load_sample("s", bad)
        assert e.value.code == "path_not_allowed"
    got = S.load_sample("s", "click.wav")
    assert got["path"].startswith(str(lib.state / "samples"))
    assert S.load_sample("s2", ".openmixing/samples/" + got["path"].rsplit("/", 1)[1])["path"] == got["path"]
    plan = {"version": "1.1", "sampler": {"s": {"path": str(out / "x.wav")}}, "clips": [{"track": "a.wav", "bars": 4}]}
    assert "path_not_allowed" in {e["code"] for e in P.validate_plan(plan, lib)}


def test_import_is_content_addressed_normalized_and_capped(lib):
    wav(lib.root / "quiet.wav", click() * 0.05)
    a, b = S.load_sample("q", "quiet.wav"), S.load_sample("q2", "quiet.wav")
    assert a["path"] == b["path"] and np.abs(sf.read(a["path"])[0]).max() == pytest.approx(0.891, abs=1e-3)
    wav(lib.root / "long.wav", np.zeros(31 * 8000, np.float32) + 0.1, 8000)
    with pytest.raises(OpenMixingError) as e:
        S.load_sample("long", "long.wav")
    assert e.value.code == "sample_too_long"
    assert S.load_sample("long", "long.wav", end_sec=2.0)["seconds"] == pytest.approx(2.0)
    wav(lib.root / "zero.wav", np.zeros(4800, np.float32))
    with pytest.raises(OpenMixingError) as e:
        S.load_sample("z", "zero.wav")
    assert e.value.code == "silent_sample"


def test_plan_round_trip_and_errors(lib):
    path = S.load_sample("hit", "click.wav")["path"]
    plan = {"version": "1.1", "name": "w", "sampler": {"hit": {"path": path, "gain_db": -2, "choke_group": "g"}},
            "clips": [{"id": "x", "track": "a.wav", "bars": 8, "triggers": [
                {"id": "s1", "kind": "sample", "slot": "hit", "at": 2, "bars": 0.5, "params": {"fit": 1, "velocity": -3}}]}]}
    m = P.apply_plan(plan, lib)
    t = m.clip("x").triggers[0]
    assert (t.kind, t.slot, t.params["fit"]) == ("sample", "hit", 1.0) and m.sampler["hit"] == SampleSlot(path, "oneshot", None, -2.0, 0.0, "g")
    out = P.export_plan(m)
    assert out["sampler"]["hit"]["choke_group"] == "g" and P.export_plan(P.apply_plan(out, lib)) == out
    back = SetModel.from_dict(m.to_dict())
    assert back.sampler == m.sampler and back.clip("x").triggers == m.clip("x").triggers
    trig = plan["clips"][0]["triggers"][0]
    cases = [({"slot": "nope"}, "bad_slot"), ({"slot": None}, "bad_slot"), ({"params": {"fit": 5}}, "out_of_range"), ({"params": {"f0": 1}}, "unknown_param")]
    for patch, code in cases:
        bad = {**plan, "clips": [{**plan["clips"][0], "triggers": [{**trig, **patch}]}]}
        assert code in {e["code"] for e in P.validate_plan(bad, lib)}, patch
    riser = {**plan, "clips": [{**plan["clips"][0], "triggers": [{"kind": "riser", "at": 4, "slot": "hit"}]}]}
    assert "bad_slot" in {e["code"] for e in P.validate_plan(riser, lib)}
    bad = {**plan, "sampler": {"hit": {"path": path, "kind": "loop", "root_bpm": 1}}}
    assert "out_of_range" in {e["code"] for e in P.validate_plan(bad, lib)}
    many = {**plan, "clips": [{**plan["clips"][0], "triggers": [{**trig, "id": f"s{i}", "bars": 1} for i in range(17)]}]}
    assert "polyphony_exceeded" in {e["code"] for e in P.validate_plan(many, lib)}
    assert "sampler" in P.plan_schema()["properties"] and "slot" in P.plan_schema()["$defs"]["clip"]["properties"]["triggers"]["items"]["properties"]


def test_tools_slots_polyphony_and_render(lib):
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    with pytest.raises(OpenMixingError) as e:
        S.trigger_sample(a, "hit", 1)
    assert e.value.code == "unknown_slot"
    S.load_sample("hit", "click.wav")
    r = S.trigger_sample(a, "hit", 2)
    assert r["ends_at_bar"] - r["starts_at_bar"] == pytest.approx(0.05 * 2 / 4, rel=0.05)
    assert S.list_samples()["slots"][0]["triggers"] == 1 and "sampler" in S.capabilities()
    with pytest.raises(OpenMixingError) as e:
        S.remove_sample("hit")
    assert e.value.code == "slot_in_use"
    with pytest.raises(OpenMixingError) as e:
        for i in range(17):
            S.trigger_sample(a, "hit", 3, 1)
    assert e.value.code == "polyphony_exceeded"
    kw = dict(info=S.info, fmt="wav", start_sec=0.0, end_sec=6.0)
    one, two = (render_set(S.cur(), out_path=lib.state / f"r{i}.wav", **kw) for i in (1, 2))
    assert np.array_equal(sf.read(one["path"])[0], sf.read(two["path"])[0])
    S.clip_of(a).triggers.clear()
    S.remove_sample("hit")
    assert S.list_samples()["slots"] == []
    assert SetModel.load(lib.set_path("t")).sampler == {}
    S.load_sample("hit", "click.wav")
    assert SetModel.load(lib.set_path("t")).sampler["hit"].kind == "oneshot"
