import librosa
import numpy as np
import pytest
from scipy.signal import stft
from scipy.stats import spearmanr
from synth import make_track

from openmixing import plan as P
from openmixing import server as S
from openmixing.core.errors import OpenMixingError
from openmixing.core.units import SR
from openmixing.dsp import synth
from openmixing.engine import render_clip
from openmixing.engine.segments import expand_window
from openmixing.library import Library
from openmixing.model import Clip, SetModel, TempoMap, Trigger
from openmixing.model.roll import roll_times

INFO = {"bpm": 120.0, "first_downbeat": 0.0, "gain_db": 0.0}


def tone(sec, hz=300.0, amp=0.3):
    y = (amp * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)
    return np.stack([y, y])


def go(clip, audio=None, win=None):
    return render_clip(clip, INFO, tone(12.0) if audio is None else audio, TempoMap(clip, 120.0), win)


def test_riser_centroid_rises_monotonically_and_level_ramps():
    y = synth.generate("riser", "r", 4 * SR, {"f0": 300.0, "f1": 6000.0})
    f, _, z = stft(y[0], SR, nperseg=2048)
    mag = np.abs(z)
    centroid = (f[:, None] * mag).sum(0) / (mag.sum(0) + 1e-12)
    assert spearmanr(np.arange(len(centroid)), centroid)[0] > 0.95 and centroid[-3] > 2 * centroid[3]
    blocks = np.sqrt((y[0, : 4 * SR // 4096 * 4096].reshape(-1, 4096) ** 2).mean(1))
    assert spearmanr(np.arange(len(blocks)), blocks)[0] > 0.99 and np.abs(y).max() == pytest.approx(1.0)


def test_downlifter_falls_and_noise_colors_differ():
    d = synth.generate("downlifter", "r", 3 * SR, {})
    assert np.abs(d[:, : SR // 2]).max() > 5 * np.abs(d[:, -SR // 2 :]).max()
    spec = {c: np.abs(np.fft.rfft(synth.generate("noise", "n", SR, {"color": c})[0])) for c in (0, 1, 2)}
    lo = lambda s: s[20:200].mean() / s[8000:12000].mean()
    assert lo(spec[0]) < lo(spec[1]) < lo(spec[2])


def test_impact_sweeps_down_and_decays():
    decay_s = 1.0
    y = synth.generate("impact", "i", 3 * SR, {"decay_s": decay_s, "noise_mix": 0.0, "f_start": 200.0, "f_end": 40.0})[0]
    f0 = np.argmax(np.abs(np.fft.rfft(y[: SR // 40]))) * SR / (SR // 40)
    f1 = np.argmax(np.abs(np.fft.rfft(y[SR // 2 : SR // 2 + SR // 5]))) * SR / (SR // 5)
    assert f0 > 1.5 * f1
    env = np.abs(y)
    peak = env[: SR // 10].max()
    tail = env[int(1.5 * decay_s * SR) : int(1.6 * decay_s * SR)].max()
    assert 20 * np.log10(tail / peak) < -40


def test_seed_from_id_is_deterministic():
    a, b = synth.generate("noise", "x", SR, {}), synth.generate("noise", "x", SR, {})
    assert a.tobytes() == b.tobytes() and not np.allclose(a, synth.generate("noise", "y", SR, {}))
    c = Clip("c", "t", 0.0, 16.0)
    c.triggers.append(Trigger("t1", "riser", 8.0, 4.0, "end", -6.0))
    assert go(c).tobytes() == go(c).tobytes()


def test_riser_end_alignment_peaks_on_target_beat_and_level_follows_deck():
    c = Clip("c", "t", 0.0, 16.0)
    c.triggers.append(Trigger("t1", "riser", 8.0, 4.0, "end", 0.0, {"tone": 1.0}))
    dry = Clip("d", "t", 0.0, 16.0)
    out, ref = go(c), go(dry)
    diff = np.abs(out - ref)[0]
    target = int(4.0 * SR)
    last = int(np.nonzero(diff > 1e-7)[0][-1])
    blocks = diff[: 4 * SR].reshape(-1, 96).max(1)
    assert abs(last - target) < 0.002 * SR and int(np.argmax(blocks)) >= len(blocks) - 2 and diff[: 2 * SR - 4].max() < 1e-6
    quiet = go(c, tone(12.0, amp=0.03))
    assert np.abs(quiet - go(dry, tone(12.0, amp=0.03)))[0].max() == pytest.approx(0.1 * diff.max(), rel=0.2)


def test_silent_deck_falls_back_to_absolute_level():
    c = Clip("c", "t", 0.0, 8.0)
    c.triggers.append(Trigger("t1", "crash", 0.0, 2.0, "start", 0.0))
    peak = np.abs(go(c, np.zeros((2, 12 * SR), np.float32))).max()
    assert peak == pytest.approx(10 ** ((-18 + 9) / 20), rel=0.02)


def test_window_render_matches_full_render():
    c = Clip("c", "t", 0.0, 24.0)
    c.triggers += [Trigger("t1", "riser", 16.0, 4.0, "end", -3.0), Trigger("t2", "impact", 16.0, 4.0, "start")]
    full = go(c)
    tm = TempoMap(c, 120.0)
    wa, wb = expand_window(c, 10.0, 18.0)
    assert wa <= 10.0 - 2.0 and wa <= 12.0 - 4.0
    na, nb = tm.beat_to_sample(wa), tm.beat_to_sample(wb)
    part = go(c, win=(na, nb))
    lo, hi = tm.beat_to_sample(10.0), tm.beat_to_sample(18.0)
    assert np.array_equal(part[:, lo - na : hi - na], full[:, lo:hi])


def test_roll_times_accelerate_and_match_rendered_onsets():
    ts = roll_times(16.0, 1.0, 0.25, "log")
    gaps = np.diff(ts)
    assert gaps[0] == pytest.approx(1.0) and gaps[-1] < 0.35 and (np.diff(gaps) < 1e-9).all()
    c = Clip("c", "t", 0.0, 16.0)
    c.triggers += [Trigger(f"s{i}", "snare", b, 0.5, "start", 0.0) for i, b in enumerate(ts)]
    y = go(c, np.zeros((2, 12 * SR), np.float32))[0]
    on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=64, units="time", backtrack=True)
    want = np.array(ts) * 0.5
    got = np.array([on[np.argmin(np.abs(on - w))] for w in want])
    assert np.abs(got - want)[1:].max() < 0.01


@pytest.fixture
def lib(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=16)
    lib = Library(tmp_path)
    lib.analyze("a.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def test_plan_round_trip_and_errors(lib):
    plan = {"version": "1.1", "name": "w", "clips": [{"id": "x", "track": "a.wav", "bars": 8, "triggers": [
        {"id": "up", "kind": "riser", "at": 4, "bars": 2, "gain_db": -3, "params": {"f1": 5000, "tone": 0.5}},
        {"id": "hit", "kind": "impact", "at": 4}]}]}
    m = P.apply_plan(plan, lib)
    up, hit = m.clip("x").triggers
    assert (up.align, up.beat, up.duration_beats, up.params["f1"]) == ("end", 16.0, 8.0, 5000.0) and hit.align == "start" and hit.start == 16.0
    out = P.export_plan(m)
    assert out["version"] == "1.1" and P.export_plan(P.apply_plan(out, lib)) == out
    assert SetModel.from_dict(m.to_dict()).clip("x").triggers == m.clip("x").triggers
    base = plan["clips"][0]["triggers"][0]
    cases = [({"kind": "zap"}, "unknown_kind"), ({"params": {"zzz": 1}}, "unknown_param"), ({"params": {"f1": 1}}, "out_of_range"),
             ({"align": "mid"}, "bad_align"), ({"gain_db": 99}, "out_of_range"), ({"at": 1}, "out_of_range"), ({"at": 99}, "out_of_range"),
             ({"extra": 1}, "unknown_field")]
    for patch, code in cases:
        bad = {**plan, "clips": [{**plan["clips"][0], "triggers": [{**base, **patch}]}]}
        assert code in {e["code"] for e in P.validate_plan(bad, lib)}, patch
    assert "triggers" in P.plan_schema()["$defs"]["clip"]["properties"]


def test_tools_trigger_and_roll_and_render(lib):
    a = S.add_clip("a.wav", bars=8, start_at_sec=0)["clip"]
    r = S.add_trigger(a, "riser", 4, 2)
    t = S.clip_of(a).triggers[0]
    assert (t.align, t.beat, r["starts_at_bar"]) == ("end", 16.0, 2.0)
    assert any("trigger" in e for e in S.get_set()["clips"][0]["events"])
    roll = S.add_roll(a, 6, 1, start_div="1/2", end_div="1/8")
    assert roll["hits"] > 4 and len(S.clip_of(a).triggers) == 1 + roll["hits"]
    S.remove_trigger(a, roll["trigger_ids"][0])
    assert len(S.clip_of(a).triggers) == roll["hits"]
    for bad in (lambda: S.add_trigger(a, "riser", 1, 4), lambda: S.add_trigger(a, "impact", 1, params={"nope": 1}),
                lambda: S.add_trigger(a, "impact", 1, gain_db=99), lambda: S.add_roll(a, 7.5, 2)):
        with pytest.raises(OpenMixingError):
            bad()
    g = S.list_generators()
    assert set(g["kinds"]) == set(synth.KINDS) and "generators" in S.capabilities()
    from openmixing.engine import render_set
    kw = dict(info=S.info, fmt="wav", start_sec=0.0, end_sec=12.0)
    one = render_set(S.cur(), out_path=lib.state / "r1.wav", **kw)
    two = render_set(S.cur(), out_path=lib.state / "r2.wav", **kw)
    import soundfile as sf
    assert np.array_equal(sf.read(one["path"])[0], sf.read(two["path"])[0])
