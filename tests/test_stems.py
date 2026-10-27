import numpy as np
import pytest

from synth import make_track
from openmixing import server as S, stems as stemlib
from openmixing.library import Library
from openmixing.setmodel import Clip


def fake_separate(audio):
    """Deterministic stand-in for Demucs: a 64-sample moving average is the bass, the rest is split between the other three."""
    k = np.ones(64, np.float32) / 64
    bass = np.stack([np.convolve(ch, k, mode="same") for ch in audio]).astype(np.float32)
    rest = audio - bass
    return {"vocals": rest * 0.3, "drums": rest * 0.3, "bass": bass, "other": rest * 0.4}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    make_track(tmp_path / "a.flac", bpm=124, bars=16)
    make_track(tmp_path / "b.flac", bpm=126, bars=16, bass_hz=61.7, pad_hz=370)
    monkeypatch.setattr(stemlib, "separate", fake_separate)
    monkeypatch.setattr(stemlib, "available", lambda: True)
    S._state.update(lib=Library(tmp_path), set=None)
    S.new_set("t")
    return tmp_path


def test_separate_caches_and_invalidates(folder):
    lib = S.lib()
    assert lib.stem_paths(lib.resolve("a.flac")) is None
    assert not lib.separate("a.flac")["cached"]
    assert lib.separate("a.flac")["cached"]
    assert set(lib.stem_paths(lib.resolve("a.flac"))) == set(stemlib.STEMS)
    (folder / "a.flac").write_bytes((folder / "a.flac").read_bytes() + b"\0")  # file changed -> stems are stale
    assert lib.stem_paths(lib.resolve("a.flac")) is None


def test_stems_clip_needs_separation(folder):
    with pytest.raises(ValueError, match="separate_stems"):
        S.add_clip("a.flac", bars=4, stems=True)


def test_clip_roundtrip_and_old_sets_load(folder):
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4, stems=True)["clip"]
    S.set_stem(c, "vocals", 1, 0.0)
    back = Clip.from_dict(S.clip_of(c).to_dict())
    assert back.stems and back.lanes["stem_vocals"].points
    d = S.clip_of(c).to_dict(); d.pop("stems")
    assert Clip.from_dict(d).stems is False  # sets saved before stems existed


def test_set_stem_requires_stems_on(folder):
    c = S.add_clip("a.flac", bars=4)["clip"]
    with pytest.raises(ValueError, match="stems=True"):
        S.set_stem(c, "vocals", 0, 0.0)


def test_render_stem_levels(folder):
    from openmixing import render as R
    from openmixing.setmodel import TempoMap
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4, stems=True)["clip"]
    clip, e = S.clip_of(c), S.info(S.clip_of(c).track)
    tm = TempoMap(clip, e["bpm"])
    cache = R.AudioCache()
    full = R.render_clip(clip, e, R.load_audio(clip, e, cache), tm)
    for stem in ("drums", "bass", "other"):
        S.set_stem(c, stem, 0, 0.0)
    only_vox = R.render_clip(clip, e, R.load_audio(clip, e, cache), tm)
    assert set(R.load_audio(clip, e, cache)) == {"vocals"}  # muted stems are never loaded
    assert 0 < np.abs(only_vox).max() < 0.5 * np.abs(full).max()


def test_stem_blend_and_audit(folder):
    for t in ("a.flac", "b.flac"):
        S.lib().separate(t)
    a = S.add_clip("a.flac", bars=16)["clip"]
    b = S.add_clip("b.flac", bars=16)["clip"]
    r = S.apply_transition(a, b, "stem_blend", bars=8, at_bar=8)
    assert S.clip_of(a).stems and S.clip_of(b).stems and r["style"] == "stem_blend"
    assert not [i for i in S.audit_set()["issues"] if i["severity"] in ("error", "warn")]


def test_stem_blend_without_stems_fails(folder):
    a = S.add_clip("a.flac", bars=16)["clip"]
    b = S.add_clip("b.flac", bars=16)["clip"]
    with pytest.raises(ValueError, match="separate_stems"):
        S.apply_transition(a, b, "stem_blend", bars=8, at_bar=8)


def test_four_stems_and_quality_report(folder):
    r = S.lib().separate("a.flac")
    assert set(r["stems"]) == {"vocals", "drums", "bass", "other"} == set(stemlib.STEMS)
    q = r["quality"]
    assert set(q["stems"]) == set(stemlib.STEMS) and q["reconstruction_match"] > 0.99
    assert abs(sum(v["energy_share"] for v in q["stems"].values()) - 1) < 0.02
    assert all(0 <= v["confidence"] <= 1 and "confidence_components" in v for v in q["stems"].values())
    assert S.lib().separate("a.flac")["cached"] is True


def test_quality_report_flags_empty_stem():
    x = np.random.default_rng(0).standard_normal((2, 48000 * 3)).astype(np.float32) * 0.1
    q = stemlib.quality_report(x, {"vocals": x * 0, "drums": x * 0.5, "bass": x * 0.2, "other": x * 0.3})
    assert q["stems"]["vocals"]["empty"] and q["stems"]["vocals"]["confidence"] == 0


def test_legacy_cache_is_detected_and_resplit(folder):
    import json
    lib = S.lib()
    t = lib.resolve("a.flac")
    d = lib.state / "stems" / t["id"]
    d.mkdir(parents=True)
    for n in ("vocals", "drums_bass", "harmonic"):
        (d / f"{n}.flac").write_bytes(b"x")
    (d / "stems.json").write_text(json.dumps({"fingerprint": lib._fingerprint(t["path"]), "model": "htdemucs"}))
    assert lib.stem_paths(t) is None and lib.stems_legacy(t)
    with pytest.raises(ValueError, match="old 3-stem layout"):
        S.add_clip("a.flac", bars=4, stems=True)
    assert lib.separate("a.flac")["cached"] is False
    assert not lib.stems_legacy(t) and not (d / "drums_bass.flac").exists()
    assert S.add_clip("a.flac", bars=4, stems=True)["stems"]


def test_legacy_set_and_plan_load(folder):
    from openmixing import plan as planlib
    from openmixing.setmodel import Clip
    c = Clip.from_dict({"id": "c1", "track": "x", "stems": True, "lanes": {
        "stem_drums_bass": [[0, 1], [4, 0]], "stem_harmonic": [[0, 0.5]], "stem_vocals": [[0, 0]]}})
    assert c.lanes["stem_drums"].points == c.lanes["stem_bass"].points == [[0, 1], [4, 0]]
    assert c.lanes["stem_other"].points == [[0, 0.5]] and c.lanes["stem_vocals"].points == [[0, 0]]
    S.lib().separate("a.flac")
    plan = {"version": planlib.PLAN_VERSION, "clips": [{"track": "a.flac", "bars": 4, "stems": True, "stem_levels": {
        "drums_bass": [{"at": 0, "level": 0}], "harmonic": [{"at": 0, "level": 0.5}]}}]}
    assert planlib.validate_plan(plan, S.lib()) == []
    m = planlib.apply_plan(plan, S.lib())
    assert m.clips[0].lanes["stem_bass"].points and m.clips[0].lanes["stem_other"].points


def test_stem_preset_instrumental_and_acapella(folder):
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4)["clip"]
    S.stem_preset(c, "instrumental")
    clip = S.clip_of(c)
    assert clip.stems and clip.lanes["stem_vocals"].sample(np.array([2.0]))[0] == 0
    assert clip.lanes["stem_bass"].sample(np.array([2.0]))[0] == 1
    S.stem_preset(c, "acapella", at_bar=1)
    assert clip.lanes["stem_vocals"].sample(np.array([8.0]))[0] == 1 and clip.lanes["stem_drums"].sample(np.array([8.0]))[0] == 0


def test_stem_blend_swaps_drums_and_bass_without_doubling(folder):
    for t in ("a.flac", "b.flac"):
        S.lib().separate(t)
    a = S.add_clip("a.flac", bars=16)["clip"]
    b = S.add_clip("b.flac", bars=16)["clip"]
    S.apply_transition(a, b, "stem_blend", bars=8, at_bar=8)
    o, i = S.clip_of(a), S.clip_of(b)
    mid = 8 * 4 + 16.0   # midpoint of the blend, in outgoing-clip beats
    assert o.lanes["stem_bass"].sample(np.array([mid + 2]))[0] == 0 and i.lanes["stem_bass"].sample(np.array([16.0 + 2]))[0] == 1


def _render(c):
    from openmixing import render as R
    from openmixing.setmodel import TempoMap
    clip, e = S.clip_of(c), S.info(S.clip_of(c).track)
    return R.render_clip(clip, e, R.load_audio(clip, e, R.AudioCache()), TempoMap(clip, e["bpm"]))


def test_stem_fx_touches_only_its_stem(folder):
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4, stems=True)["clip"]
    dry = _render(c)
    S.add_fx(c, "echo", 1, 2, beats="1/2", stem="vocals")
    wet = _render(c)
    assert np.abs(wet - dry).max() > 1e-3
    for stem in ("drums", "bass", "other"):
        S.set_stem(c, stem, 0, 0.0)
    only_vox_wet = _render(c)
    S.clip_of(c).fx.clear()
    only_vox_dry = _render(c)
    assert np.abs(only_vox_wet - only_vox_dry).max() > 1e-3
    S.set_stem(c, "vocals", 0, 0.0)  # vocals muted: an echo on them adds nothing, so the other stems really were dry
    S.add_fx(c, "echo", 1, 2, beats="1/2", stem="vocals")
    muted_wet = _render(c)
    S.clip_of(c).fx.clear()
    assert np.abs(muted_wet - _render(c)).max() < 1e-6


def test_stem_fx_tail_rings_past_the_clip(folder):
    from openmixing import render as R
    S.lib().separate("a.flac")
    c = S.add_clip("a.flac", bars=4, stems=True)["clip"]
    S.add_fx(c, "echo", 3, 1, beats="1/2", stem="vocals")
    buf = _render(c)
    assert np.abs(buf[:, buf.shape[1] - R.TAIL:]).max() > 1e-4


def test_stem_fx_needs_stems(folder):
    c = S.add_clip("a.flac", bars=4)["clip"]
    with pytest.raises(Exception, match="stems_off|plays the full mix"):
        S.add_fx(c, "echo", 1, 2, stem="vocals")
    S.clip_of(c).fx.append(__import__("openmixing.setmodel", fromlist=["FxEvent"]).FxEvent("f", "echo", 4.0, 4.0, 1.0, 1.0, {}, "vocals"))
    with pytest.raises(ValueError, match="stem fx"):
        _render(c)


def test_stem_fx_in_plans(folder):
    from openmixing import plan as P
    S.lib().separate("a.flac")
    def plan(stems, stem):
        return {"version": "1.0", "name": "p", "clips": [{"id": "x", "track": "a.flac", "src_start": 0, "bars": 8, "stems": stems,
                "fx": [{"fx": "echo", "at": 2, "bars": 2, "stem": stem}]}]}
    assert P.validate_plan(plan(True, "vocals"), S.lib()) == []
    assert [e["code"] for e in P.validate_plan(plan(True, "guitar"), S.lib())] == ["unknown_stem"]
    assert [e["code"] for e in P.validate_plan(plan(False, "vocals"), S.lib())] == ["stems_off"]
    m = P.apply_plan(plan(True, "vocals"), S.lib()) if hasattr(P, "apply_plan") else None
    assert P.export_plan(m)["clips"][0]["fx"][0]["stem"] == "vocals"


def fake_separate_808(audio):
    """Demucs on 808 music: the sub bass ends up in the drums stem and the bass stem is silent."""
    s = fake_separate(audio)
    s["drums"] = s["drums"] + s["bass"]
    s["bass"] = np.zeros_like(s["bass"])
    return s


def test_bass_carrier_helper():
    assert stemlib.bass_carrier({}) == "bass" and stemlib.low_stem({}) == "bass"
    info = {"stem_quality": {"stems": {"bass": {"empty": True}}}}
    assert stemlib.bass_carrier(info) == "drums_lowband" and stemlib.low_stem(info) == "drums"
    assert stemlib.bass_carrier({"stem_quality": {"stems": {"bass": {"empty": False}}}}) == "bass"


def test_effective_reports_carrier(folder, monkeypatch):
    S.lib().separate("a.flac")
    assert S.lib().effective("a.flac")["bass_carrier"] == "bass"
    monkeypatch.setattr(stemlib, "separate", fake_separate_808)
    S.lib().separate("b.flac")
    e = S.lib().effective("b.flac")
    assert e["stem_quality"]["stems"]["bass"]["empty"] and e["bass_carrier"] == "drums_lowband"


def _drums_level(c, beat):
    return float(S.clip_of(c).lanes["stem_drums"].sample(np.array([beat]))[0])


def test_stem_blend_moves_drums_to_the_bass_swap_when_bass_is_empty(folder, monkeypatch):
    monkeypatch.setattr(stemlib, "separate", fake_separate_808)
    for t in ("a.flac", "b.flac"):
        S.lib().separate(t)
    a = S.add_clip("a.flac", bars=16)["clip"]
    b = S.add_clip("b.flac", bars=16)["clip"]
    r = S.apply_transition(a, b, "stem_blend", bars=8, at_bar=8)
    a0 = 8 * 4
    assert _drums_level(a, a0 + 0.45 * 32) == 1.0          # default would already be fading at 40%
    assert _drums_level(a, a0 + 0.5 * 32 + 4.5) == 0.0
    assert any("808" in n for n in r["notes"])


def test_stem_blend_default_keeps_early_drum_swap(folder):
    for t in ("a.flac", "b.flac"):
        S.lib().separate(t)
    a = S.add_clip("a.flac", bars=16)["clip"]
    b = S.add_clip("b.flac", bars=16)["clip"]
    S.apply_transition(a, b, "stem_blend", bars=8, at_bar=8)
    assert _drums_level(a, 8 * 4 + 0.45 * 32) < 1.0


def _overlap_issues(folder, sep, monkeypatch):
    monkeypatch.setattr(stemlib, "separate", sep)
    for t in ("a.flac", "b.flac"):
        S.lib().separate(t)
    a = S.add_clip("a.flac", bars=8, start_at_sec=0, stems=True)["clip"]
    b = S.add_clip("b.flac", bars=8, start_at_sec=0, stems=True)["clip"]
    S.set_stem(b, "bass", 0, 0.0)  # the DJ kills the "bass" stem on the incoming deck
    return [i for i in S.audit_set()["issues"] if "bass clash" in i["message"]]


def test_audit_bass_clash_reads_the_bass_stem_normally(folder, monkeypatch):
    assert not _overlap_issues(folder, fake_separate, monkeypatch)


def test_audit_bass_clash_reads_drums_when_bass_is_empty(folder, monkeypatch):
    assert _overlap_issues(folder, fake_separate_808, monkeypatch)


def test_stems_survive_an_analysis_version_bump(folder, monkeypatch):
    from openmixing import library
    S.lib().separate("a.flac")
    monkeypatch.setattr(library, "ANALYSIS_VERSION", library.ANALYSIS_VERSION + 1)
    assert S.lib().stem_paths(S.lib().resolve("a.flac")) is not None
