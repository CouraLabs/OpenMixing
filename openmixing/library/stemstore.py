from __future__ import annotations

import os

import soundfile as sf

from ..analysis import stems as stemlib
from ..core.decode import decode
from ..core.jsonio import read_json, write_json
from ..core.units import SR


class StemStore:
    def stem_paths(self, t: dict) -> dict[str, str] | None:
        d = self.state / "stems" / t["id"]
        meta = self._stem_meta(t)
        if not meta or meta.get("version") != stemlib.CACHE_VERSION:
            return None
        paths = {n: str(d / f"{n}.flac") for n in stemlib.STEMS}
        same_file = str(meta.get("fingerprint", "")).split(":")[:2] == self._fingerprint(t["path"]).split(":")[:2]
        ok = same_file and all(os.path.exists(p) for p in paths.values())
        return paths if ok else None

    def _stem_meta(self, t: dict) -> dict | None:
        return read_json(self.state / "stems" / t["id"] / "stems.json", strict=True)

    def stems_legacy(self, t: dict) -> bool:
        meta = self._stem_meta(t)
        return bool(meta) and meta.get("version") != stemlib.CACHE_VERSION

    def separate(self, ref: str, force: bool = False) -> dict:
        t = self.resolve(ref)
        if not force and (paths := self.stem_paths(t)):
            return {"track": t["file"], "cached": True, "stems": paths, "quality": (self._stem_meta(t) or {}).get("quality")}
        audio = decode(t["path"])
        out = stemlib.separate(audio)
        quality = stemlib.quality_report(audio, out)
        d = self.state / "stems" / t["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "stems.json").unlink(missing_ok=True)
        for old in stemlib.LEGACY_STEMS:
            (d / f"{old}.flac").unlink(missing_ok=True)
        for name, y in out.items():
            sf.write(d / f"{name}.flac", y.T, SR, format="FLAC", subtype="PCM_24")
        write_json(d / "stems.json", {"fingerprint": self._fingerprint(t["path"]), "model": stemlib.MODEL,
                                      "version": stemlib.CACHE_VERSION, "quality": quality})
        return {"track": t["file"], "cached": False, "stems": self.stem_paths(t), "quality": quality}
