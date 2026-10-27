import numpy as np
import pytest

from openmixing import candidates as C
from openmixing import plan as P
from openmixing.library import Library
from synth import make_track


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("cand")
    make_track(d / "a.wav", bpm=124, bars=48)
    make_track(d / "b.wav", bpm=126, bars=48, bass_hz=65)
    make_track(d / "far.wav", bpm=90, bars=48, bass_hz=70)
    lib = Library(d)
    for n in ("a.wav", "b.wav", "far.wav"):
        lib.analyze(n)
    return lib


def test_windows_cover_both_sides(lib):
    f = C.track_facts(lib, "a.wav")
    for side in ("exit", "entry"):
        ws = C.find_windows(f, side)
        assert ws and all(0 <= w["start_bar"] < w["end_bar"] <= f["n_bars"] and 0 <= w["confidence"] <= 1 for w in ws)
        assert ws == sorted(ws, key=lambda w: -w["score"])
    with pytest.raises(ValueError):
        C.find_windows(f, "middle")


def test_pair_report_measures_everything(lib):
    r = C.pair_report(C.track_facts(lib, "a.wav"), C.track_facts(lib, "b.wav"))
    assert r["tempo"]["stretch_percent"] == pytest.approx((124 / 126 - 1) * 100, abs=0.5)
    assert abs(r["key"]["semitones_to_match"]) <= 6
    assert r["vocal"]["risk"] is None and r["vocal"]["confidence"] == 0
    for k in ("tempo", "key", "loudness", "low_end", "energy", "phrase"):
        assert "confidence" in r[k]
    far = C.pair_report(C.track_facts(lib, "a.wav"), C.track_facts(lib, "far.wav"))
    assert abs(far["tempo"]["stretch_percent"]) > 8 or far["tempo"]["ratio_used"] != 1.0


def test_key_shift():
    assert C.key_shift_semitones("8B", "8B") == 0 and C.key_shift_semitones("8A", "8B") == 0
    assert C.key_shift_semitones("8B", "9B") == 5   # G major up 5 semitones is C major
    assert C.key_shift_semitones("8B", "7B") == -5


def test_candidates_are_valid_plans_with_subscores(lib):
    r = C.suggest(lib, "a.wav", "b.wav", limit=20)
    assert r["candidates"] and r["n_feasible"] >= len(r["candidates"])
    assert [c["total"] for c in r["candidates"]] == sorted((c["total"] for c in r["candidates"]), reverse=True)
    for c in r["candidates"]:
        assert P.validate_plan(c["plan"], lib) == []
        assert set(c["sub_scores"]) >= {"tempo", "harmony", "energy_flow", "bass_clash", "vocal_clash", "phrase"}
        assert all(0 <= v <= 1 for v in c["sub_scores"].values()) and 0 <= c["confidence"] <= 1
    m = P.apply_plan(r["candidates"][0]["plan"], lib)
    assert [x.id for x in m.clips] == ["out", "in"]
    styles = {c["style"] for c in r["candidates"]}
    assert len(styles) >= 3


def test_infeasible_techniques_have_reasons(lib):
    r = C.suggest(lib, "a.wav", "b.wav", limit=50)
    assert not any(c["style"] == "stem_blend" for c in r["candidates"])
    assert any(x["style"] == "stem_blend" and "stems" in x["reason"] for x in r["infeasible"])


def test_goals_change_ranking(lib):
    smooth = C.suggest(lib, "a.wav", "b.wav", "smooth")["candidates"][0]
    dramatic = C.suggest(lib, "a.wav", "b.wav", {"dramatic": 1})["candidates"][0]
    assert smooth["style"] != dramatic["style"] or smooth["total"] != dramatic["total"]
    with pytest.raises(ValueError):
        C.resolve_goals(["chill"])


def test_vocal_activity():
    sr, bpm = 8000, 120
    y = np.zeros(sr * 8, np.float32)
    y[: sr * 2] = 0.5 * np.sin(np.arange(sr * 2) / 10)
    v = C.vocal_activity(y, sr, 0.0, bpm, 4)
    assert v[0] == 1.0 and v[2] == 0.0


def _facts(n, ph, secs):
    sections = [{"label": l, "start_bar": a, "end_bar": b, "bars": b - a, "energy": 0.5, "bass": 0.5} for l, a, b in secs]
    return {"e": {"bpm": 124.0}, "d": {"mix_points": {"mix_out": {"bar": 0}, "mix_in": {"bar": 0}}}, "n_bars": n, "sections": sections,
            "phrase": ph, "energy": [0.5] * n, "bass": [0.5] * n, "vocal": None, "tempo_conf": 1.0, "key_conf": 1.0}


def test_exit_never_starts_in_first_phrase_and_entry_never_in_outro():
    f = _facts(96, 16, [("intro", 0, 16), ("groove", 16, 32), ("drop", 32, 64), ("breakdown", 64, 80), ("outro", 80, 96)])
    ex, en = C.find_windows(f, "exit"), C.find_windows(f, "entry")
    assert ex and all(w["label"] != "intro" and w["start_bar"] >= 16 for w in ex)
    assert en and all(w["label"] != "outro" and w["start_bar"] < 80 for w in en)
    assert any(w["label"] == "intro" for w in en) and any(w["label"] == "outro" for w in ex)


def test_exit_mix_out_cue_cannot_pull_window_into_first_phrase():
    f = _facts(96, 16, [("drop", 0, 40), ("breakdown", 40, 56), ("outro", 56, 96)])
    assert all(w["start_bar"] >= 16 for w in C.find_windows(f, "exit"))


def test_short_track_keeps_its_windows():
    f = _facts(24, 16, [("intro", 0, 8), ("drop", 8, 24)])
    assert C.find_windows(f, "exit") and C.find_windows(f, "entry")
