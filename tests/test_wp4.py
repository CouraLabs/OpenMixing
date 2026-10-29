import numpy as np
import pytest
from synth import make_track

from openmixing import plan as P
from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.units import SR
from openmixing.dsp import dynamics
from openmixing.dsp import fx as fxm
from openmixing.engine import render_clip
from openmixing.library import Library
from openmixing.model import Clip, DuckEvent, FxEvent, SetModel, TempoMap

INFO = {"bpm": 120.0, "first_downbeat": 0.0, "gain_db": 0.0}


def sine(sec, hz=500.0, amp=0.5):
    y = (amp * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)
    return np.stack([y, y])


def db(x):
    return 20 * np.log10(np.abs(x).max() + 1e-9)


def test_compressor_static_curve_matches_transfer_function():
    for amp_db in (-30, -20, -12, -6):
        x = sine(1.5, amp=10 ** (amp_db / 20))
        y = dynamics.compress(x, -20.0, 4.0, 5.0, 50.0, knee_db=0.0)
        over = amp_db + 20
        want = amp_db if over <= 0 else amp_db - over * (1 - 1 / 4)
        assert db(y[:, SR:]) == pytest.approx(want, abs=0.3)


def test_compressor_soft_knee_starts_below_threshold_and_makeup_adds_gain():
    x = sine(1.5, amp=0.1)
    hard = db(dynamics.compress(x, -20.0, 4.0, 5.0, 50.0, knee_db=0.0)[:, SR:])
    soft = db(dynamics.compress(x, -20.0, 4.0, 5.0, 50.0, knee_db=12.0)[:, SR:])
    made = db(dynamics.compress(x, -20.0, 4.0, 5.0, 50.0, knee_db=0.0, makeup_db=6.0)[:, SR:])
    assert hard == pytest.approx(db(x), abs=0.2) and soft < db(x) - 0.5 and made == pytest.approx(hard + 6, abs=0.3)


def test_attack_time_constant():
    key = np.concatenate([np.zeros(SR // 2), np.ones(SR)]).astype(np.float32)
    x = np.ones((2, len(key)), np.float32)
    y = dynamics.duck(x, key, 12.0, 50.0, 100.0, -40.0)[0]
    red = -20 * np.log10(y)
    t63 = np.argmax(red >= 0.632 * 12.0) - SR // 2
    assert t63 / SR * 1000 == pytest.approx(50.0, rel=0.15)


def kicks(sec, bpm=120.0):
    n = int(sec * SR)
    y = np.zeros(n, np.float32)
    step = int(60 / bpm * SR)
    m = int(0.05 * SR)
    for i in range(0, n - m, step):
        y[i : i + m] = np.sin(2 * np.pi * 60 * np.arange(m) / SR) * np.exp(-np.arange(m) / (0.02 * SR))
    return np.stack([y, y]), step


def test_duck_depth_recovery_and_silence_outside_window():
    k, step = kicks(4.0)
    x = sine(4.0, 300.0, 0.3)
    y = dynamics.duck(x, k, 9.0, 3.0, 100.0, -30.0)
    onset = 2 * step + int(0.03 * SR)
    assert db(x[:, onset : onset + 200]) - db(y[:, onset : onset + 200]) >= 8.0
    late = 2 * step + int(0.42 * SR)
    assert db(x[:, late : late + 400]) - db(y[:, late : late + 400]) < 1.0
    clip = Clip("c", "t", 0.0, 8.0)
    tm = TempoMap(clip, 120.0)
    full = sine(5.0, 300.0, 0.3)
    ev = DuckEvent("d", "c", "x", 4.0, 2.0, 9.0, -30.0, 3.0, 100.0)
    key = np.zeros(int(4.0 * SR), np.float32)
    key[:] = 1.0
    ref = render_clip(clip, INFO, full, tm)
    out = render_clip(clip, INFO, full, tm, ducks=[(ev, key)])
    assert np.allclose(out[:, : 2 * SR - 10], ref[:, : 2 * SR - 10]) and np.allclose(out[:, 3 * SR + 400 : 4 * SR], ref[:, 3 * SR + 400 : 4 * SR])
    mid = slice(2 * SR + SR // 2, 2 * SR + 9 * SR // 10)
    assert db(ref[:, mid]) - db(out[:, mid]) == pytest.approx(9.0, abs=1.0)


def test_sidechain_pump_period_is_one_beat():
    x = np.ones((2, 4 * SR), np.float32)
    y = dynamics.pump(x, 0.5, 9.0, 0.8, 1.0)[0]
    g = y - y.mean()
    ac = np.correlate(g[: 2 * SR], g[: 2 * SR], "full")[2 * SR - 1 :]
    lag = np.argmax(ac[SR // 8 :]) + SR // 8
    assert lag / SR == pytest.approx(0.5, abs=0.01)
    assert y.min() < 0.4 and y.max() > 0.95


def test_fx_catalog_has_dynamics_with_renderers():
    assert {"compressor", "sidechain_pump"} <= set(fxm.CATALOG) <= set(fxm.RENDERERS)


def chain_clip(steps):
    clip = Clip("c", "t", 0.0, 8.0)
    clip.fx.append(FxEvent("f", "chain", 0.0, 8.0, 1.0, 1.0, {}, None, steps))
    return clip


def test_chain_order_matters_and_render_is_deterministic():
    audio = sine(5.0, 440.0, 0.4)
    crush = {"fx": "bitcrush", "beats": 1.0, "wet": 1.0, "params": {"bits": 4}}
    verb = {"fx": "reverb", "beats": 1.0, "wet": 0.5, "params": {}}

    def go(steps):
        c = chain_clip(steps)
        return render_clip(c, INFO, audio, TempoMap(c, 120.0))

    a, b = go([crush, verb]), go([verb, crush])
    assert np.isfinite(a).all() and np.isfinite(b).all() and not np.allclose(a, b)
    assert go([crush, verb]).tobytes() == a.tobytes()


def test_wet_curve_fades_the_effect_in():
    audio = sine(5.0, 440.0, 0.4)
    c = Clip("c", "t", 0.0, 8.0)
    c.fx.append(FxEvent("f", "distortion", 0.0, 8.0, 1.0, 1.0, {"drive_db": 30}, None, None, [[0.0, 0.0], [8.0, 1.0]]))
    dry = Clip("d", "t", 0.0, 8.0)
    y, ref = (render_clip(k, INFO, audio, TempoMap(k, 120.0))[0] for k in (c, dry))
    assert np.abs(y[: SR // 20] - ref[: SR // 20]).max() < 0.03 and np.abs(y[3 * SR : 4 * SR] - ref[3 * SR : 4 * SR]).max() > 0.1


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    make_track(tmp_path / "b.wav", bpm=120, bars=16)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    lib.analyze("b.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def test_plan_chain_wet_points_ducks_round_trip_and_errors(lib):
    plan = {"version": "1.1", "name": "w", "clips": [
        {"id": "x", "track": "a.wav", "bars": 8, "fx": [
            {"id": "ch", "at": 2, "bars": 4, "wet": 0.8, "chain": [{"fx": "compressor", "params": {"ratio": 6}}, {"fx": "echo", "beats": "1/2", "wet": 0.4}],
             "wet_points": [[0, 0], [2, 1]]}]},
        {"id": "y", "track": "b.wav", "bars": 8, "start": {"over": "x", "at_bar": 0}}],
        "ducks": [{"id": "dk", "target": "y", "key": "x", "at": 1, "bars": 4, "depth_db": 8}]}
    m = P.apply_plan(plan, lib)
    f = m.clip("x").fx[0]
    assert f.fx == "chain" and [s["fx"] for s in f.chain] == ["compressor", "echo"] and f.wet_curve == [[0.0, 0.0], [8.0, 1.0]]
    assert m.ducks[0].key == "x" and m.ducks[0].beat == 4.0 and m.ducks[0].depth_db == 8.0
    out = P.export_plan(m)
    assert out["version"] == "1.1" and out["ducks"][0]["target"] == "y" and out["clips"][0]["fx"][0]["chain"][1]["beats"] == "1/2"
    assert P.export_plan(P.apply_plan(out, lib)) == out
    assert SetModel.from_dict(m.to_dict()).ducks == m.ducks and SetModel.from_dict(m.to_dict()).clip("x").fx[0].chain == f.chain
    fx = plan["clips"][0]["fx"][0]
    cases = [({**fx, "fx": "echo"}, "fx_or_chain"), ({**fx, "chain": [{"fx": "nope"}]}, "unknown_fx"),
             ({**fx, "chain": [{"fx": "echo"}] * 7}, "out_of_range"), ({**fx, "wet_points": [[2, 0], [1, 1]]}, "not_ascending"),
             ({**fx, "chain": [{"fx": "echo", "params": {"zzz": 1}}]}, "unknown_param")]
    for bad_fx, code in cases:
        bad = {**plan, "clips": [{**plan["clips"][0], "fx": [bad_fx]}, plan["clips"][1]]}
        assert code in {e["code"] for e in P.validate_plan(bad, lib)}, code
    for d, code in (({"key": "self:drums"}, "stems_off"), ({"depth_db": 99}, "out_of_range"), ({"target": "zz"}, "unknown_clip"),
                    ({"key": "x:vocals"}, "unknown_clip")):
        bad = {**plan, "ducks": [{**plan["ducks"][0], **d}]}
        assert code in {e["code"] for e in P.validate_plan(bad, lib)}, code


def test_tools_add_fx_chain_and_ducks(lib):
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    b = S.add_clip("b.wav", bars=8, over_clip=a, at_bar=0)["clip"]
    r = S.add_fx(a, "echo", 1, 2, chain=[{"fx": "bitcrush"}, {"fx": "reverb", "beats": 2, "wet": 0.5}], wet_points=[[0, 0], [1, 1]])
    e = S.clip_of(a).fx[0]
    assert e.id == r["fx_id"] and e.fx == "chain" and e.chain[1]["beats"] == 2.0 and e.wet_curve == [[0.0, 0.0], [4.0, 1.0]]
    with pytest.raises(ValueError):
        S.add_fx(a, "echo", 1, 2, chain=[{"fx": "nope"}])
    with pytest.raises(ValueError):
        S.add_fx(a, "echo", 1, 2, wet_points=[[2, 0], [1, 1]])
    d = S.add_duck(b, a, 1, 4, depth_db=10)["duck_id"]
    assert S.cur().ducks[0].key == a and "ducked 10 dB" in S.get_set()["ducks"][0]
    with pytest.raises(OpenMixingError):
        S.add_duck(b, "self:drums", 0, 2, stem="vocals")
    with pytest.raises(OpenMixingError):
        S.add_duck(b, "zzz", 0, 2)
    S.remove_duck(d)
    assert not S.cur().ducks
    S.add_duck(b, a, 0, 2)
    S.remove_clip(b)
    assert not S.cur().ducks


def test_cross_deck_duck_lowers_target_only_in_window(lib):
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    b = S.add_clip("b.wav", bars=8, over_clip=a, at_bar=0)["clip"]
    S.set_volume(a, 0, 0.0)
    S.set_volume(a, 4, 1.0)
    S.set_volume(a, 4.01, 1.0)
    from openmixing.engine import render_set
    kw = dict(info=S.info, out_path=lib.state / "r.wav", fmt="wav", start_sec=0.0, end_sec=16.0)
    ref = render_set(S.cur(), **{**kw, "out_path": lib.state / "ref.wav"})
    S.add_duck(b, a, 2, 4, depth_db=20, threshold_db=-40)
    out = render_set(S.cur(), **kw)
    import soundfile as sf
    x, _ = sf.read(ref["path"])
    y, _ = sf.read(out["path"])
    win = slice(int(8.2 * SR), int(11.5 * SR))
    assert np.sqrt((y[win] ** 2).mean()) < np.sqrt((x[win] ** 2).mean()) * 0.8
    assert np.allclose(y[: int(3.9 * SR)], x[: int(3.9 * SR)], atol=1e-4)


def test_self_keyed_stem_duck_lowers_only_the_target_stem(tmp_path, monkeypatch):
    from openmixing import engine as R
    from openmixing.analysis import stems as stemlib
    from test_stems import fake_separate

    make_track(tmp_path / "a.flac", bpm=124, bars=16)
    monkeypatch.setattr(stemlib, "separate", fake_separate)
    monkeypatch.setattr(stemlib, "available", lambda: True)
    S.STATE.update(lib=Library(tmp_path), set=None)
    S.new_set("t")
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4, stems=True)["clip"]
    S.set_stem(c, "bass", 0, 0.0)
    S.set_stem(c, "other", 0, 0.0)

    def go():
        clip, e = S.clip_of(c), S.info(S.clip_of(c).track)
        return R.render_clip(clip, e, R.load_audio(clip, e, R.AudioCache()), TempoMap(clip, e["bpm"]), ducks=[(d, None) for d in S.cur().ducks])

    dry = go()
    S.add_duck(c, "self:drums", 0, 4, depth_db=12, threshold_db=-60, stem="vocals")
    wet = go()
    assert np.abs(wet).max() < np.abs(dry).max() and not np.allclose(wet, dry)
    assert S.cur().ducks[0].key == "self:drums" and S.cur().ducks[0].stem == "vocals"
