import asyncio

import numpy as np
import pytest
import soundfile as sf
from synth import make_track

from openmixing import server as S
from openmixing import validate as V
from openmixing.analysis import stems as stemlib
from openmixing.library import Library
from openmixing.model import SetModel
from openmixing.validate.gap import silent_gaps
from openmixing.validate.harmony import dissonance


def fake_separate(audio):
    k = np.ones(64, np.float32) / 64
    bass = np.stack([np.convolve(ch, k, mode="same") for ch in audio]).astype(np.float32)
    rest = audio - bass
    return {"vocals": rest * 0.3, "drums": rest * 0.3, "bass": bass, "other": rest * 0.4}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    make_track(tmp_path / "a.wav", bpm=124, bars=16, pad_hz=330, vocal_bars=range(16))
    make_track(tmp_path / "b.wav", bpm=124, bars=16, pad_hz=330 * 2 ** (6 / 12), vocal_bars=range(16))
    t = np.arange(int(16 * 4 * 60 / 124 * 44100)) / 44100
    for n, semi in (("p0", 0), ("p6", 6), ("p7", 7), ("p0b", 0)):
        y = (0.3 * np.sin(2 * np.pi * 330 * 2 ** (semi / 12) * t)).astype(np.float32)
        sf.write(tmp_path / f"{n}.wav", np.stack([y, y], 1), 44100, subtype="PCM_24")
    monkeypatch.setattr(stemlib, "separate", fake_separate)
    monkeypatch.setattr(stemlib, "available", lambda: True)
    S.STATE.update(lib=Library(tmp_path), set=None)
    for n in ("p0", "p6", "p7", "p0b"):
        S.override_track(f"{n}.wav", bpm=124.0, first_downbeat_sec=0.0)
    S.new_set("t")
    yield tmp_path
    S.STATE.update(lib=None, set=None)


def overlap_set(a, b, stems=False):
    S.new_set("o")
    ca = S.add_clip(a, bars=8, stems=stems)["clip"]
    cb = S.add_clip(b, bars=8, over_clip=ca, at_bar=4, stems=stems)["clip"]
    return ca, cb


def validation():
    return asyncio.run(S.render(format="wav"))["validation"]["checks"]


def test_version_bumped():
    assert V.VERSION == 3


def test_vocal_collision_measured_on_stems(folder):
    S.lib().separate("a.wav")
    S.lib().separate("b.wav")
    ca, cb = overlap_set("a.wav", "b.wav", stems=True)
    vc = validation()["vocal_collision"]
    assert vc["method"] == "measured"
    assert vc["seconds_overlap"] == pytest.approx(4 * 4 * 60 / 124, abs=0.5)
    assert vc["segments"][0]["clips"] == [ca, cb] and vc["max_level_db"] > -40
    S.set_stem(cb, "vocals", 0.0, 0)
    vc = validation()["vocal_collision"]
    assert vc["seconds_overlap"] == 0.0 and vc["segments"] == []


def test_vocal_collision_estimated_without_stems(folder):
    overlap_set("a.wav", "b.wav")
    vc = validation()["vocal_collision"]
    assert vc["method"] == "estimated" and set(vc["methods"].values()) == {"estimated"}
    assert 0.0 <= vc["confidence"] <= 1.0


def test_vocal_collision_function():
    db = {"a": np.array([-60, -20, -20, -20, -20, -60.0]), "b": np.array([-60, -60, -20, -20, -60, -60.0])}
    r = V.vocal_collision(db, {"a": "measured", "b": "estimated"}, {"a": 1.0, "b": 0.4}, 0.5, 10.0)
    assert r["seconds_overlap"] == 1.0 and r["segments"][0]["start_sec"] == 11.0 and r["method"] == "mixed" and r["confidence"] == 0.4
    assert V.vocal_collision({"a": db["a"]}, {"a": "measured"}, {"a": 1.0}, 0.5, 0)["seconds_overlap"] == 0.0


def test_dissonance_ordering_by_interval():
    def pc(i):
        v = np.zeros(12)
        v[i] = 1.0
        return v

    d = [dissonance(pc(0), pc(k)) for k in (0, 7, 2, 6, 1)]
    assert d == sorted(d) and d[0] == 0.0 and d[-1] > d[-2] > d[-3]


def test_harmonic_clash_tritone_vs_fifth_and_same_key(folder):
    out = {}
    for name, other in (("tritone", "p6.wav"), ("fifth", "p7.wav"), ("same", "p0b.wav")):
        overlap_set("p0.wav", other)
        out[name] = validation()["harmonic_clash"]
    assert out["tritone"]["method"] == "chroma_roughness" and out["tritone"]["status"] == "measured"
    assert out["tritone"]["mean_dissonance"] > out["fifth"]["mean_dissonance"] > out["same"]["mean_dissonance"] - 1e-3
    assert out["same"]["key_distance_max"] == 0


def test_harmonic_clash_not_applicable_without_overlap(folder):
    S.new_set("solo")
    S.add_clip("a.wav", bars=4)
    h = validation()["harmonic_clash"]
    assert h["status"] == "not_applicable" and h["mean_dissonance"] is None and h["confidence"] == 0.0


def test_tension_match_rising_vs_flat(tmp_path):
    make_track(tmp_path / "r.wav", bpm=120, bars=32, riser_bars=(0, 32))
    make_track(tmp_path / "f.wav", bpm=120, bars=32)
    out = {}
    for n in ("r", "f"):
        y, sr = sf.read(tmp_path / f"{n}.wav", dtype="float32")
        out[n] = V.tension_match(V.rendered_tension(y.mean(1), sr), 0.0, 1.0, {"shape": "rising"})
    assert out["r"]["correlation"] >= 0.6 and out["r"]["correlation"] > out["f"]["correlation"]
    assert out["f"]["confidence"] < out["r"]["confidence"]
    assert out["r"]["agreement"] > 0.5


def test_tension_targets():
    t = V.tension_target("build_release", 10)
    assert t[:8].argmax() == 7 and t[8] == pytest.approx(0.1)
    assert t[-1] == pytest.approx(0.1) and V.tension_target("falling", 5)[0] == 1.0
    r = V.tension_match(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), 0.0, 1.0, {"points": [[0, 0], [5, 1]]})
    assert r["correlation"] == pytest.approx(1.0, abs=1e-3) and r["max_error"] < 0.01
    flat = V.tension_match(np.full(6, 0.5), 0.0, 1.0, {"shape": "rising"})
    assert flat["correlation"] == 0.0


def test_silent_gap_measured():
    sr = 8000
    y = np.sin(2 * np.pi * 200 * np.arange(sr * 8) / sr).astype(np.float32)
    y[int(4 * sr) : int(4.5 * sr)] = 0
    g = silent_gaps(y, sr, 10.0)
    assert len(g) == 1 and g[0]["start_sec"] == pytest.approx(14.0, abs=0.06) and g[0]["end_sec"] == pytest.approx(14.5, abs=0.06)


def test_intent_keys_parse_and_report():
    good = {"vocal_overlap_max_sec": 2, "vocal_clash_intended": False, "harmonic_dissonance_max": 0.4, "harmonic_clash_intended": True,
            "pre_drop_gap_min_beats": 1, "tension_target": {"shape": "rising", "min_correlation": 0.7}}
    assert set(good) <= set(V.INTENT_KEYS)
    from openmixing.plan.context import Ctx
    from openmixing.plan.intent import parse_intent
    cx = Ctx()
    assert parse_intent(cx, good) == good and not cx.errors
    cx = Ctx()
    parse_intent(cx, {"harmonic_dissonance_max": 2, "tension_target": {"shape": "zigzag"}, "vocal_overlap_max_sec": -1})
    assert {e["field"] for e in cx.errors} >= {"plan.intent.harmonic_dissonance_max", "plan.intent.tension_target.shape",
                                                "plan.intent.vocal_overlap_max_sec"}
    cx = Ctx()
    parse_intent(cx, {"tension_target": {"shape": "rising", "points": [[0, 0], [1, 1]]}})
    assert cx.errors
    checks = {"vocal_collision": {"seconds_overlap": 6.0, "method": "measured"},
              "harmonic_clash": {"mean_dissonance": 0.5, "confidence": 0.9},
              "tension_match": {"correlation": 0.5, "target": "rising", "confidence": 0.8},
              "pre_drop_gap": [{"beats": 2.0}], "beat_alignment": [], "low_end_overlap": {"overlap_sec": 0}, "clipping": {"true_peak_dbtp": -2}}
    rep = {r["item"]: r["status"] for r in V.intent_report({**good, "vocal_clash_intended": True}, checks)}
    assert rep == {"vocal_overlap_max_sec": "fail", "vocal_clash_intended": "pass", "harmonic_dissonance_max": "fail",
                   "harmonic_clash_intended": "pass", "tension_target": "fail", "pre_drop_gap_min_beats": "pass"}


def test_summary_and_compare_include_new_metrics():
    c = {"clipping": {"true_peak_dbtp": -2, "clipped_samples": 0}, "unplanned_silence": {"regions": []}, "beat_alignment": [],
         "loudness_continuity": [], "low_end_overlap": {"overlap_sec": 0}, "vocal_collision": {"seconds_overlap": 3.0},
         "harmonic_clash": {"mean_dissonance": 0.2}}
    a = V.summarize({"checks": c, "blocking_failures": []})
    b = dict(a, vocal_overlap_sec=0.0, harmonic_dissonance_mean=0.4)
    d = V.compare(a, b)
    assert d["vocal_overlap_sec"]["lower"] == "b" and d["harmonic_dissonance_mean"]["lower"] == "a"


def test_pre_drop_gap_and_tension_in_render(folder):
    S.STATE["set"] = SetModel("g")
    S.STATE["set"].intent = {"tension_target": {"shape": "rising", "min_correlation": 0.0}, "pre_drop_gap_min_beats": 0.5}
    S.add_clip("a.wav", bars=8)
    r = asyncio.run(S.render(format="wav"))["validation"]
    assert "tension_match" in r["checks"] and "pre_drop_gap" in r["checks"]
    assert {i["item"] for i in r["intent_report"]} == {"tension_target", "pre_drop_gap_min_beats"}
