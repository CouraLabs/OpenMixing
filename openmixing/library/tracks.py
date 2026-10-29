from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf

from ..analysis import online, vocals
from ..analysis import stems as stemlib
from ..analysis.features import analyze_audio
from ..analysis.meta import format_report, probe, read_tags
from ..core.decode import decode
from ..core.jsonio import read_json, write_json
from ..core.units import out_bars, to_bars

ANALYSIS_VERSION = 5


class TrackStore:
    def _f(self, tid: str, kind: str = "") -> Path:
        return self.state / "analysis" / f"{tid}{kind}.json"

    @staticmethod
    def _fingerprint(path: str) -> str:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}:{ANALYSIS_VERSION}"

    def cached(self, t: dict) -> dict | None:
        d = read_json(self._f(t["id"]), {})
        return d if d.get("fingerprint") == self._fingerprint(t["path"]) else None

    def _grid_stems(self, t: dict, separate: bool) -> dict | None:
        if separate and stemlib.available():
            self.separate(t["id"])
        paths = self.stem_paths(t)
        if not paths:
            return None
        quality = ((self._stem_meta(t) or {}).get("quality") or {}).get("stems") or {}
        return {k: sf.read(paths[k], always_2d=True)[0].T for k in ("drums", "bass") if not quality.get(k, {}).get("empty")}

    def analyze(self, ref: str, force: bool = False, use_stems: bool = False) -> dict:
        t = self.resolve(ref)
        d = self.cached(t)
        if d and not force and not (use_stems and not d["tempo"]["stems_used"] and stemlib.available()):
            return d
        pr = probe(t["path"])
        fmt = format_report(t["path"], pr)
        audio = decode(t["path"])
        feats = analyze_audio(audio, t["path"], pr.get("bitrate_kbps"), self._grid_stems(t, use_stems))
        wave = feats.pop("_waveform")
        d = {"id": t["id"], "file": t["file"], "path": t["path"], "fingerprint": self._fingerprint(t["path"]),
             "file_info": {**pr, "format_report": fmt}, "tags": read_tags(t["path"]), **feats}
        write_json(self._f(t["id"]), d)
        write_json(self._f(t["id"], ".wave"), wave)
        return d

    def waveform(self, tid: str) -> dict | None:
        return read_json(self._f(tid, ".wave"))

    def online(self, ref: str, providers=None, artist=None, title=None, refresh=False) -> dict:
        t = self.resolve(ref)
        f = self._f(t["id"], ".online")
        if f.exists() and not refresh and not (providers or artist or title):
            return read_json(f)
        d = self.analyze(t["id"])
        res = online.lookup(t["path"], Path(t["file"]).stem, d["tags"], providers, artist, title, d["tempo"]["bpm"])
        if not (providers or artist or title) or not f.exists():
            write_json(f, res)
        return res

    def cached_online(self, tid: str) -> dict | None:
        return read_json(self._f(tid, ".online"))

    def effective(self, ref: str) -> dict:
        t = self.resolve(ref)
        d = self.analyze(t["id"])
        ov = self.overrides().get(t["id"], {})
        tempo, key, loud = d["tempo"], d["key"], d["loudness"]
        eff = {"id": t["id"], "file": t["file"], "path": t["path"],
               "bpm": float(ov.get("bpm", tempo["bpm"])), "first_downbeat": float(ov.get("first_downbeat", tempo["first_downbeat_sec"])),
               "gain_db": float(ov.get("gain_db", loud["suggested_gain_db"])), "duration": float(d["duration_sec"]),
               "camelot": ov.get("camelot", key["camelot"]), "key": ov.get("key", key["key"]),
               "key_conf": 1.0 if ("camelot" in ov or "key" in ov) else round(key["confidence"], 3),
               "tuning_cents": float(key.get("tuning_cents", 0.0)),
               "energy": d["energy"]["level_1_10"], "overrides": ov, "stems": self.stem_paths(t),
               "stems_legacy": self.stems_legacy(t)}
        eff["stem_quality"] = (self._stem_meta(t) or {}).get("quality") if eff["stems"] else None
        eff["n_bars"] = int(tempo["n_bars"]) if "bpm" not in ov else int((eff["duration"] - eff["first_downbeat"]) * eff["bpm"] / 240)
        vb = vocals.proxy_block(d)
        eff["vocal_curve"], eff["vocal_conf"] = vb["curve"], vb["confidence"]
        eff["bass_carrier"] = stemlib.bass_carrier(eff)
        return eff

    def to_bars(self, ref: str, value, field: str = "position") -> float:
        e = self.effective(ref)
        d = self.analyze(e["id"])
        cues = {k: v["bar"] for k, v in self.cues(e["id"]).items()}
        return to_bars(value, e, cues, d["structure"]["sections"], field)

    def out_bars(self, clip, value, field: str = "position") -> float:
        return out_bars(value, clip.src_start_beat / 4.0, lambda v: self.to_bars(clip.track, v, field), field)
