import asyncio
import subprocess

import numpy as np
import pytest
from synth import make_track

from openmixing import plan as P
from openmixing import server as S
from openmixing import validate as V
from openmixing.core.errors import OpenMixingError
from openmixing.library import Library
from openmixing.model import SetModel


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("var")
    make_track(d / "a.wav", bpm=124, bars=32)
    make_track(d / "b.wav", bpm=124, bars=32, bass_hz=65)
    lib = Library(d)
    lib.analyze("a.wav")
    lib.analyze("b.wav")
    S.STATE.update(lib=lib, set=SetModel("t"))
    yield lib
    S.STATE.update(lib=None, set=None)


def plan(intent=None, offset_ms=0):
    p = {"version": "1.0", "name": "x", "clips": [
        {"id": "out", "track": "a.wav", "bars": 16},
        {"id": "in", "track": "b.wav", "bars": 16, "start": {"over": "out", "at_bar": 8, "offset_ms": offset_ms},
         "eq": {"low": {"points": [[0, -80], [8, -80], [8.1, 0]]}}}]}
    if intent is not None:
        p["intent"] = intent
    return p


def test_intent_validated_and_round_trips(lib):
    good = {"beat_drift_ms_max": 10, "loudness_jump_lu": [-2, 2], "silence_intended": [[1, 2]], "low_end_clash_intended": True}
    m = P.apply_plan(plan(good), lib)
    assert m.intent["loudness_jump_lu"] == [-2.0, 2.0]
    assert P.export_plan(m)["intent"] == m.intent
    assert SetModel.from_dict(m.to_dict()).intent == m.intent
    errs = P.validate_plan(plan({"loudness_jump_lu": [2, -2], "nope": 1, "low_end_clash_intended": "yes"}), lib)
    fields = {e["field"] for e in errs}
    assert {"plan.intent.loudness_jump_lu", "plan.intent.low_end_clash_intended"} <= fields and any("nope" in e["message"] for e in errs)
    assert "intent" in P.plan_schema()["properties"]


def checks(drift=0.0, jumps=(0.5,), lo=0.0, gaps=()):
    return {"beat_alignment": [{"max_drift_ms": drift}], "loudness_continuity": [{"jump_lu": j} for j in jumps],
            "planned_gaps": list(gaps), "low_end_overlap": {"overlap_sec": lo}, "clipping": {"true_peak_dbtp": -1.5}}


def test_intent_report_pass_fail():
    rep = lambda i, **k: {r["item"]: r["status"] for r in V.intent_report(i, checks(**k))}
    assert rep({"beat_drift_ms_max": 5}, drift=2) == {"beat_drift_ms_max": "pass"}
    assert rep({"beat_drift_ms_max": 5}, drift=40) == {"beat_drift_ms_max": "fail"}
    assert rep({"beat_drift_intended": True}, drift=40)["beat_drift_intended"] == "pass"
    assert rep({"beat_drift_intended": True}, drift=1)["beat_drift_intended"] == "fail"       # declared but did not happen
    assert rep({"low_end_clash_intended": True}, lo=6)["low_end_clash_intended"] == "pass"    # intended clash happened
    assert rep({"low_end_clash_intended": True}, lo=0)["low_end_clash_intended"] == "fail"
    assert rep({"low_end_overlap_max_sec": 2}, lo=6)["low_end_overlap_max_sec"] == "fail"
    assert rep({"loudness_jump_lu": [-1, 1]}, jumps=(0.4, -0.9))["loudness_jump_lu"] == "pass"
    assert rep({"loudness_jump_lu": [-1, 1]}, jumps=(3,))["loudness_jump_lu"] == "fail"
    gap = {"start_sec": 5.0, "end_sec": 6.0, "planned": True}
    assert rep({"silence_intended": [[5.2, 5.8]]}, gaps=[gap])["silence_intended"] == "pass"
    assert rep({"silence_intended": [[20, 21]]}, gaps=[gap])["silence_intended"] == "fail"


def test_low_end_overlap_counts_simultaneous_bass():
    a = np.full(40, -20.0)
    b = np.full(40, -120.0)
    b[10:30] = -25.0
    r = V.low_end_overlap({"a": a, "b": b}, 0.25, 100.0)
    assert r["overlap_sec"] == 5.0 and r["regions"][0]["start_sec"] == 102.5 and r["regions"][0]["clips"] == ["a", "b"]
    assert V.low_end_overlap({"a": a}, 0.25, 0)["overlap_sec"] == 0.0


def test_compare_marks_lower_without_judging():
    base = {"integrated_lufs": -14.0, "true_peak_dbtp": -1.0, "clipped_samples": 0, "unplanned_silence_sec": 0.0, "max_drift_ms": 2.0,
            "max_abs_jump_lu": 1.0, "low_end_overlap_sec": 4.0, "intent_pass": 1, "intent_fail": 0, "blocking": []}
    other = dict(base, max_drift_ms=30.0, low_end_overlap_sec=4.0, integrated_lufs=-12.5, blocking=["clipping"])
    d = V.compare(base, other)
    assert d["max_drift_ms"] == {"a": 2.0, "b": 30.0, "delta": 28.0, "lower": "a"}
    assert d["low_end_overlap_sec"]["lower"] == "tie" and "lower" not in d["integrated_lufs"]
    assert d["blocking"] == {"a": [], "b": ["clipping"]}


def test_render_variations_and_compare(lib):
    good, bad = plan({"beat_drift_ms_max": 10, "low_end_overlap_max_sec": 1}), plan({"beat_drift_ms_max": 10}, offset_ms=60)
    r = asyncio.run(S.render_variations("a.wav", "b.wav", plans=[good, bad], bars_before=2, bars_after=2))
    t0, t1 = r["takes"]
    assert t0["max_drift_ms"] < 2 and t1["max_drift_ms"] == pytest.approx(60, abs=2)
    assert t0["intent_fail"] == 0 and t1["intent_fail"] == 1 and t0["blocking"] == []
    assert S.cur().clips == []                                  # the current set is untouched
    c = S.compare_renders(t0["id"], t1["id"], detail=True)
    assert c["diff"]["max_drift_ms"]["lower"] == "a" and c["diff"]["intent_fail"]["delta"] == 1
    assert c["intent_report"]["b"][0]["status"] == "fail"
    with pytest.raises(OpenMixingError) as e:
        S.compare_renders(t0["id"], "nothing")
    assert e.value.code == "not_found"
    bad_plan = {"version": "1.0", "clips": [{"track": "a.wav", "bars": -1}]}
    r2 = asyncio.run(S.render_variations("a.wav", "b.wav", plans=[bad_plan]))
    assert "error" in r2["takes"][0]


def test_render_defaults_to_mp3_320_and_flac_on_request(lib):
    S.STATE["set"] = SetModel("fmt")
    S.add_clip("a.wav", bars=4)
    r = asyncio.run(S.render())
    assert r["path"].endswith("fmt.mp3") and r["format"] == "mp3"
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,bit_rate", "-of", "csv=p=0", r["path"]],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out.startswith("mp3") and abs(int(out.split(",")[1]) - 320000) < 2000
    assert np.isfinite(r["metrics"]["integrated_lufs"])
    f = asyncio.run(S.render(format="flac"))
    assert f["path"].endswith("fmt.flac")
