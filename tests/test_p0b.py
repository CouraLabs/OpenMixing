import asyncio
import json
import os

import numpy as np
import pytest
from synth import make_track

from openmixing import candidates
from openmixing import server as S
from openmixing.analysis import region as R
from openmixing.analysis.confidence import combine, unit
from openmixing.core.errors import OpenMixingError
from openmixing.library import ANALYSIS_VERSION, Library
from openmixing.server.versions import versions


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("p0b")
    make_track(d / "a.wav", bpm=124, bars=48)
    make_track(d / "b.wav", bpm=126, bars=48, bass_hz=65)
    lib = Library(d)
    lib.analyze("a.wav")
    lib.analyze("b.wav")
    return lib


@pytest.fixture
def server(lib):
    S.STATE.update(lib=lib, set=None)
    yield S
    S.STATE.update(lib=None, set=None)


def _walk(o, path=""):
    if isinstance(o, dict):
        if "confidence" in o:
            yield path, o
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")


# ---------------------------------------------------------------- confidence
def test_confidence_helpers():
    assert unit(5, 0, 10) == 0.5 and unit(-1, 0, 1) == 0.0 and unit(2, 0, 1) == 1.0
    assert combine({"a": 1.0, "b": 0.5, "c": None})[0] == 0.75
    assert combine({"a": 0.5, "b": 0.5}, "product")[0] == 0.25
    assert combine({})[0] == 0.0


def test_every_estimate_has_confidence_and_components(lib):
    d = lib.analyze("a.wav")
    assert d["fingerprint"].endswith(f":{ANALYSIS_VERSION}") and ANALYSIS_VERSION >= 2
    for blk in ("tempo", "key", "loudness", "energy", "structure"):
        assert 0 <= d[blk]["confidence"] <= 1 and d[blk]["confidence_components"], blk
    assert 0 <= d["tempo"]["downbeat_confidence"] <= 1 and d["tempo"]["downbeat_confidence_components"]
    assert "margin" in d["key"] and "camelot" in d["key"]  # old field names kept
    assert d["structure"]["sections"]
    for s in d["structure"]["sections"]:
        assert 0 <= s["confidence"] <= 1 and set(s["confidence_components"]) <= {"boundary_contrast", "label_margin"}
    mp = d["mix_points"]
    for blk in (mp["mix_in"], mp["mix_out"], *mp["cue_points"]):
        assert 0 <= blk["confidence"] <= 1 and blk["confidence_components"]
    for c in d["structure"]["loop_candidates"]:
        assert 0 <= c["confidence"] <= 1
    assert len(d["structure"]["onset_curve"]) == len(d["structure"]["energy_curve"])


def test_confidence_reflects_evidence(lib):
    d = lib.analyze("a.wav")
    t = d["tempo"]
    parts = t["confidence_components"]
    assert t["confidence"] == pytest.approx(np.mean(list(parts.values())), abs=0.01)
    assert d["key"]["confidence"] == pytest.approx(np.mean(list(d["key"]["confidence_components"].values())), abs=0.01)


def test_vocal_confidence_from_stem(lib, tmp_path):
    t = lib.resolve("a.wav")
    stem = tmp_path / "vocals.wav"
    sr = 48000
    x = np.zeros(sr * 60, np.float32)
    x[: sr * 30] = 0.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 30) / sr)
    import soundfile as sf
    sf.write(stem, x, sr)
    e = {"id": t["id"] + "v", "bpm": 120.0, "first_downbeat": 0.0, "stems": {"vocals": str(stem)}}
    ve = candidates.vocal_entry(lib, e, 30)
    assert 0 < ve["confidence"] <= 1 and set(ve["confidence_components"]) == {"stem_level", "contrast"}
    assert ve["curve"][0] > 0.9 and ve["curve"][-1] == 0
    assert candidates.vocal_entry(lib, {**e, "stems": None}, 30) is None


# ---------------------------------------------------------------- versions
def test_versions_in_responses(server):
    v = versions()
    assert v["analysis"] == ANALYSIS_VERSION and v["plan_schema"] and v["validation"] and v["candidates"]
    server.new_set("vtest")
    assert server.get_set()["versions"] == v
    assert server.find_windows("a.wav", "exit")["versions"] == v
    assert server.pair_report("a.wav", "b.wav")["versions"] == v
    assert server.suggest_transitions("a.wav", "b.wav", limit=1, include_plan=False)["versions"] == v
    assert server.capabilities()["versions"] == v
    assert server.find_windows("a.wav", "exit")["vocal"]["available"] is False


# ---------------------------------------------------------------- regions
def test_get_region_slices_and_defaults(server, lib):
    d = lib.analyze("a.wav")
    r = server.get_region("a.wav", 8, 16)
    assert r["start_bar"] == 8 and r["end_bar"] == 16 and r["grid"] == "analysis"
    assert r["energy"]["bars"] == list(range(8, 16))
    assert r["energy"]["values"] == d["structure"]["energy_curve"][8:16]
    assert set(r) >= {"energy", "bass", "sections", "versions"} and "beats" not in r
    assert all(s["end_bar"] > 8 and s["start_bar"] < 16 for s in r["sections"])
    assert server.get_region("a.wav", 0)["end_bar"] == 16  # default length


def test_get_region_fields_units_and_downsampling(server, lib):
    r = server.get_region("a.wav", "32 beats", "64 beats", what=["onsets", "beats", "loops", "cues", "waveform", "vocal"])
    assert r["start_bar"] == 8 and r["end_bar"] == 16
    assert r["beats"]["n_beats"] == 32 and r["beats"]["times_sec"][0] == pytest.approx(lib.effective("a.wav")["first_downbeat"] + 8 * 240 / 124, abs=0.01)
    assert r["vocal"]["available"] is False and len(r["waveform"]["peak"]) > 0
    assert len(r["onsets"]["values"]) == 8
    r2 = server.get_region("a.wav", 0, 40, what=["energy"], max_points=8)
    assert len(r2["energy"]["values"]) == 8 and r2["energy"]["step_bars"] == 5
    server.set_cue("a.wav", "hook", 4)
    assert server.get_region("a.wav", "cue:hook", "cue:hook+4", what=["cues"])["cues"]["hot"]["hook"]["bar"] == 4


def test_get_region_errors(server):
    with pytest.raises(OpenMixingError) as ei:
        server.get_region("a.wav", 10, 5)
    assert ei.value.code == "invalid_range" and ei.value.field == "end"
    with pytest.raises(OpenMixingError) as ei:
        server.get_region("a.wav", 0, 8, what=["nope"])
    assert ei.value.field == "what" and "energy" in ei.value.fix


def test_region_maps_overridden_grid(lib):
    e = lib.effective("a.wav")
    d = lib.analyze("a.wav")
    e2 = {**e, "bpm": e["bpm"] * 2, "first_downbeat": e["first_downbeat"]}
    r = R.slice_region(d, e2, 0, 8, ("energy",))
    assert r["grid"] == "override"
    assert r["energy"]["values"] == d["structure"]["energy_curve"][:4]  # 8 double-time bars = 4 original bars


def test_analyze_track_region_filter(server, lib):
    run = asyncio.run
    full = run(server.analyze_track("a.wav"))
    part = run(server.analyze_track("a.wav", region=[8, "section:drop"] if any(s["label"] == "drop" for s in full["structure"]["sections"]) else [8, 24]))
    st = part["structure"]
    assert "energy_curve" not in st and "sections" not in st and st["region"]["energy"]["values"]
    assert len(json.dumps(part)) < len(json.dumps(full))
    assert full["versions"] == part["versions"] and part["region"][0] == 8
    with pytest.raises(OpenMixingError):
        run(server.analyze_track("a.wav", region=[5]))


# ---------------------------------------------------------------- file rules
def test_inputs_must_stay_in_tracks_folder(lib, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    make_track(outside / "evil.wav", bpm=120, bars=4)
    for ref in (str(outside / "evil.wav"), "../evil.wav", "/etc/passwd", "~/x.wav"):
        with pytest.raises(OpenMixingError) as ei:
            lib.resolve(ref)
        assert ei.value.code == "path_not_allowed" and ei.value.field == "track" and ei.value.fix
    assert lib.resolve(str(lib.root / "a.wav"))["file"] == "a.wav"
    assert lib.resolve("./a.wav")["file"] == "a.wav"


def test_symlink_out_of_folder_is_not_a_track(tmp_path_factory):
    d, outside = tmp_path_factory.mktemp("inside"), tmp_path_factory.mktemp("elsewhere")
    make_track(outside / "x.wav", bpm=120, bars=4)
    make_track(d / "ok.wav", bpm=120, bars=4)
    os.symlink(outside / "x.wav", d / "link.wav")
    assert [t["file"] for t in Library(d).scan()] == ["ok.wav"]


def test_outputs_stay_in_state_folder(lib):
    for bad in ("../x.flac", "/tmp/x.flac", lib.root / "x.flac"):
        with pytest.raises(OpenMixingError) as ei:
            lib.output_path(bad, "out")
        assert ei.value.code == "path_not_allowed" and ei.value.field == "out"
    p = lib.render_path("../../evil/../set", "flac")
    assert p.parent == (lib.state / "renders").resolve() and "/" not in p.name
    assert lib.render_path("....hidden", "wav").name == "hidden.wav"
    assert lib.set_path("../../x").parent == (lib.state / "sets").resolve()


def test_ui_audio_route_rejects_traversal(lib):
    import urllib.error
    import urllib.request

    from openmixing.player import start
    (lib.state / "renders" / "s.flac").write_bytes(b"fLaC")
    (lib.root / "secret.wav").write_bytes(b"RIFF")
    ui = start(lib, _free_port())
    url = ui["url"].rstrip("/")
    assert urllib.request.urlopen(f"{url}/audio/s.flac").status in (200, 206)
    for p in ("/audio/..%2F..%2Fsecret.wav", "/audio/../secret.wav"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url + p)
        assert ei.value.code == 404


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- capabilities
def test_capabilities(server, lib):
    c = server.capabilities()
    assert {"flac", "wav", "mp3"} <= set(c["output_formats"]) and "wav" in c["input_formats"]
    assert c["default_output_format"] == "mp3" and c["mp3_bitrate_kbps"] == 320
    assert "echo" in c["fx"] and "bass_swap" in c["transition_styles"] and "linear" in c["curves"]
    assert c["stems"]["names"] == ["vocals", "drums", "bass", "other"] and c["max_lanes"] is None
    assert c["file_rules"] == {"read": str(lib.root), "write": str(lib.state)}
    assert "torch" in c["optional"] and "eq_low" in c["mixer_lanes"]
    json.dumps(c)
