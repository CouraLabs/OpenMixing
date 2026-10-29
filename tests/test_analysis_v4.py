import asyncio

import numpy as np
import pytest
import soundfile as sf
from scipy.stats import spearmanr
from synth import make_track

from openmixing.analysis import vocals
from openmixing.analysis.structure import structure
from openmixing.analysis.tension import FPS, lead_ins
from openmixing.library import ANALYSIS_VERSION, Library

VOCAL_BARS = list(range(8, 12)) + list(range(20, 24))


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("v4")
    make_track(d / "v.wav", bpm=120, bars=32, vocal_bars=VOCAL_BARS, riser_bars=(16, 32))
    return Library(d)


def test_version_bumped_and_old_cache_ignored(lib):
    assert ANALYSIS_VERSION == 5
    t = lib.resolve("v.wav")
    d = lib.analyze("v.wav")
    assert lib.cached(t) is not None
    assert lib.cached({**t, "path": t["path"]}) and d["fingerprint"].endswith(":5")
    from openmixing.core.jsonio import write_json
    write_json(lib._f(t["id"]), {**d, "fingerprint": d["fingerprint"][:-1] + "4"})
    assert lib.cached(t) is None
    lib.analyze("v.wav", True)


def test_new_blocks_have_confidence(lib):
    d = lib.analyze("v.wav")
    for k in ("tension", "harmony"):
        assert 0 <= d[k]["confidence"] <= 1 and d[k]["confidence_components"]
    assert len(d["tension"]["curve"]) == len(d["structure"]["energy_curve"]) == len(d["harmony"]["chroma_bar"])
    assert set(d["tension"]["weights"]) == {"density", "brightness", "rise", "bass_absence", "noise"}
    assert all(len(r) == 12 for r in d["harmony"]["chroma_bar"])
    assert d["harmony"]["key_by_phrase"]


def test_proxy_vocal_is_low_confidence_and_tracks_vocal_bars(lib):
    d = lib.analyze("v.wav")
    v = vocals.vocal_block(lib, d, lib.effective("v.wav"))
    assert v["source"] == "proxy" and v["confidence"] <= 0.4
    c = np.array(v["curve"])
    quiet = [k for k in range(len(c)) if all(abs(k - b) > 1 for b in VOCAL_BARS) and k < 16]
    assert c[VOCAL_BARS].mean() > c[quiet].mean()


def test_stem_vocal_curve_and_regions(tmp_path):
    sr, bpm = 8000, 120
    y = np.zeros(sr * 2 * 16, np.float32)
    for b in (4, 5, 6, 7, 10):
        y[b * 2 * sr : (b + 1) * 2 * sr] = 0.4 * np.sin(np.arange(2 * sr) / 9)
    sf.write(tmp_path / "vocals.wav", y, sr)

    class L:
        state = tmp_path

    (tmp_path / "analysis").mkdir()
    e = {"id": "x", "bpm": bpm, "first_downbeat": 0.0, "stems": {"vocals": str(tmp_path / "vocals.wav")}, "n_bars": 16}
    v = vocals.vocal_block(L, {}, e)
    c = np.array(v["curve"])
    assert v["source"] == "stem" and (c[[4, 5, 6, 7, 10]] > 0.5).all() and (c[[0, 1, 2, 13, 14]] < 0.5).all()
    assert [(r["start_bar"], r["end_bar"]) for r in v["regions"]] == [(4, 8), (10, 11)]
    assert vocals.regions([0, 0.5, 0, 0.5, 0, 0, 0, 0.5]) == [
        {"start_bar": 1, "end_bar": 4, "mean": pytest.approx(0.333, abs=1e-3)}, {"start_bar": 7, "end_bar": 8, "mean": 0.5}]
    assert vocals.vocal_block(L, {}, {**e, "stems": None})["source"] is None


def test_tension_rises_through_the_riser(lib):
    d = lib.analyze("v.wav")
    curve = np.array(d["tension"]["curve"])[16:31]
    assert spearmanr(np.arange(len(curve)), curve)[0] > 0.8


def test_lead_in_gap_measures_planted_silence():
    fps, bar_sec, beat = FPS, 2.0, 0.5
    n = int(10 * bar_sec * fps)
    S = np.ones((8, n))
    a, b = int((3 * bar_sec - beat) * fps), int(3 * bar_sec * fps)
    S[:, a:b] = 1e-4
    secs = [{"label": "intro", "start_bar": 0}, {"label": "drop", "start_bar": 3}, {"label": "drop", "start_bar": 1}]
    lead_ins(S, secs, 0.0, bar_sec, beat)
    assert abs(secs[1]["lead_in"]["gap_beats"] - 1.0) < 0.25 and secs[1]["lead_in"]["gap_depth_db"] > 30
    assert "lead_in" not in secs[0] and "lead_in" not in secs[2]


def test_extra_bar_columns_do_not_change_sections():
    rng = np.random.default_rng(1)
    n = 48
    X = np.zeros((n, 84))
    lv = np.repeat([1.0, 2.0, 3.0], 16)
    X[:, 0], X[:, 1] = -30 + 10 * lv, -35 + 12 * (lv % 2)
    X[:, 4:68] = lv[:, None] * rng.standard_normal(64)[None] + 0.01 * rng.standard_normal((n, 64))
    a = structure({"n_bars": n, "X": X, "bar_sec": 1.9}, 0.5)
    X2 = X.copy()
    X2[:, 80:] = 1000 * rng.standard_normal((n, 4))
    b = structure({"n_bars": n, "X": X2, "bar_sec": 1.9}, 0.5)
    assert a["sections"] == b["sections"]


def test_tools_expose_new_blocks(lib):
    from openmixing import server as S
    S.STATE.update(lib=lib, set=None)
    try:
        r = asyncio.run(S.analyze_track("v.wav", max_points=8))
        assert len(r["tension"]["curve"]["values"]) <= 8 and "chroma_bar" not in r["harmony"] and r["vocal"]["source"] == "proxy"
        assert "tension" not in asyncio.run(S.analyze_track("v.wav", include_tension=False))
        g = S.get_region("v.wav", 0, 8, ["tension", "harmony"])
        assert len(g["tension"]["values"]) == 8 and len(g["harmony"]["chroma"]) == 8 and len(g["harmony"]["chroma"][0]) == 12
    finally:
        S.STATE.update(lib=None, set=None)


def test_stems_refine_grid_and_snap_bpm(tmp_path):
    from synth import make_track as mk

    from openmixing.analysis.features import analyze_audio
    from openmixing.core.decode import decode
    mk(tmp_path / "s.wav", bpm=123.0, bars=32, lead_delay=0.21)
    audio = decode(str(tmp_path / "s.wav"))
    plain = analyze_audio(audio)["tempo"]
    stemmed = analyze_audio(audio, stems={"drums": audio, "bass": audio * 0.1})["tempo"]
    assert plain["stems_used"] is False and stemmed["stems_used"] is True
    assert stemmed["bpm"] == pytest.approx(123.0, abs=0.01)
    assert stemmed["grid_source"] == "drums_stem" and 0 <= stemmed["beat_strength"] <= 1
