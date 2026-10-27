"""The track folder: discovery, stable ids, cached analysis, manual overrides, and the set / render stores."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import soundfile as sf

from .analysis.analyze import AUDIO_EXT
from .analysis.features import analyze_audio
from .analysis.meta import format_report, probe, read_tags
from . import tags as taglib
from . import stems as stemlib
from .analysis import online
from .decode import decode
from .models import SR
from .errors import OpenMixingError
from .setmodel import SetModel

ANALYSIS_VERSION = 3


class Library:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("OPENMIXING_DIR") or Path.cwd()).expanduser().resolve()
        self.state = self.root / ".openmixing"
        for d in ("analysis", "sets", "renders", "stems"):
            (self.state / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- discovery
    def scan(self) -> list[dict]:
        out, seen, cache = [], {}, self._hashes()
        dirty = False
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".") and ".openmixing" not in p.parts and self._inside(p, self.root):
                rel = str(p.relative_to(self.root))
                st = p.stat()
                key = f"{st.st_mtime_ns}:{st.st_size}"
                hit = cache.get(rel)
                if not hit or hit["key"] != key:
                    hit = cache[rel] = {"key": key, "id": self.content_id(p)}
                    dirty = True
                tid = hit["id"]
                if tid in seen:  # identical audio under another name: keep ids unique, flag the copy
                    tid = f"{tid}-{hashlib.blake2b(rel.encode(), digest_size=2).hexdigest()}"
                    out.append({"id": tid, "file": rel, "path": str(p), "duplicate_of": seen[hit["id"]]})
                    continue
                seen[tid] = rel
                out.append({"id": tid, "file": rel, "path": str(p)})
        if dirty:
            self._hash_file().write_text(json.dumps(cache))
        return out

    # ---------------------------------------------------------------- file rules
    @staticmethod
    def _inside(path: Path | str, base: Path) -> bool:
        p = Path(path).resolve()
        return p == base or base in p.parents

    def _allowed(self, path: Path | str, base: Path, field: str, what: str, where: str) -> Path:
        p = Path(path).expanduser()
        p = (p if p.is_absolute() else base / p).resolve()
        if not self._inside(p, base):
            raise OpenMixingError("path_not_allowed", f"{str(path)!r} is outside {where}; {what} must stay inside it", field,
                                f"use a file name or a path relative to {base}")
        return p

    def input_path(self, path: Path | str, field: str = "path") -> Path:
        """Resolve an input path; only the tracks folder is readable (symlinks that leave it are rejected)."""
        return self._allowed(path, self.root, field, "inputs", "the tracks folder")

    def output_path(self, path: Path | str, field: str = "path") -> Path:
        """Resolve an output path; only .openmixing/ is writable."""
        return self._allowed(path, self.state.resolve(), field, "outputs", "the .openmixing/ state folder")

    @staticmethod
    def safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).lstrip(".") or "unnamed"

    def render_path(self, name: str, fmt: str) -> Path:
        return self.output_path(self.state / "renders" / f"{self.safe_name(name)}.{fmt}", "name")

    def _hash_file(self) -> Path:
        return self.state / "hashes.json"

    def _hashes(self) -> dict:
        try:
            return json.loads(self._hash_file().read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def content_id(path: Path | str) -> str:
        """Stable id from the file bytes, so renaming or moving a file keeps its analysis, stems and set references."""
        h = hashlib.blake2b(digest_size=6)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def resolve(self, ref: str) -> dict:
        """Track by id, relative path, or unique case-insensitive substring of the file name."""
        tracks = self.scan()
        for t in tracks:
            if ref in (t["id"], t["file"], t["path"]):
                return t
        if os.sep in ref or ref.startswith(("~", ".")):
            p = self.input_path(ref, "track")
            if (hit := next((t for t in tracks if Path(t["path"]).resolve() == p), None)):
                return hit
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        hits = [t for t in tracks if norm(ref) in norm(t["file"])]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"no track matches {ref!r}; call list_tracks")
        raise KeyError(f"{ref!r} is ambiguous: {[t['file'] for t in hits[:8]]}")

    # ---------------------------------------------------------------- analysis cache
    def _f(self, tid: str, kind: str = "") -> Path:
        return self.state / "analysis" / f"{tid}{kind}.json"

    @staticmethod
    def _fingerprint(path: str) -> str:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}:{ANALYSIS_VERSION}"

    def cached(self, t: dict) -> dict | None:
        try:
            d = json.loads(self._f(t["id"]).read_text())
        except (OSError, ValueError):
            return None
        return d if d.get("fingerprint") == self._fingerprint(t["path"]) else None

    def analyze(self, ref: str, force: bool = False) -> dict:
        t = self.resolve(ref)
        if not force and (d := self.cached(t)):
            return d
        pr = probe(t["path"])
        fmt = format_report(t["path"], pr)
        audio = decode(t["path"])
        feats = analyze_audio(audio, t["path"], pr.get("bitrate_kbps"))
        wave = feats.pop("_waveform")
        d = {"id": t["id"], "file": t["file"], "path": t["path"], "fingerprint": self._fingerprint(t["path"]),
             "file_info": {**pr, "format_report": fmt}, "tags": read_tags(t["path"]), **feats}
        self._f(t["id"]).write_text(json.dumps(d))
        self._f(t["id"], ".wave").write_text(json.dumps(wave))
        return d

    def waveform(self, tid: str) -> dict | None:
        try:
            return json.loads(self._f(tid, ".wave").read_text())
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------- stems
    def stem_paths(self, t: dict) -> dict[str, str] | None:
        """{stem: flac path} when the track's stems are cached and still match the file, else None."""
        d = self.state / "stems" / t["id"]
        meta = self._stem_meta(t)
        if not meta or meta.get("version") != stemlib.CACHE_VERSION:
            return None
        paths = {n: str(d / f"{n}.flac") for n in stemlib.STEMS}
        same_file = str(meta.get("fingerprint", "")).split(":")[:2] == self._fingerprint(t["path"]).split(":")[:2]  # mtime + size; stems don't depend on the analysis version
        ok = same_file and all(os.path.exists(p) for p in paths.values())
        return paths if ok else None

    def _stem_meta(self, t: dict) -> dict | None:
        try:
            return json.loads((self.state / "stems" / t["id"] / "stems.json").read_text())
        except (OSError, ValueError):
            return None

    def stems_legacy(self, t: dict) -> bool:
        """True when the track only has stems from the old 3-stem layout (vocals / drums_bass / harmonic)."""
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
        (d / "stems.json").unlink(missing_ok=True)  # a half-written set is never trusted
        for old in stemlib.LEGACY_STEMS:
            (d / f"{old}.flac").unlink(missing_ok=True)
        for name, y in out.items():
            sf.write(d / f"{name}.flac", y.T, SR, format="FLAC", subtype="PCM_24")
        (d / "stems.json").write_text(json.dumps({"fingerprint": self._fingerprint(t["path"]), "model": stemlib.MODEL,
                                                  "version": stemlib.CACHE_VERSION, "quality": quality}))
        return {"track": t["file"], "cached": False, "stems": self.stem_paths(t), "quality": quality}

    # ---------------------------------------------------------------- overrides (manual corrections)
    def overrides(self) -> dict:
        try:
            return json.loads((self.state / "overrides.json").read_text())
        except (OSError, ValueError):
            return {}

    def set_override(self, tid: str, **kv) -> dict:
        o = self.overrides()
        cur = o.setdefault(tid, {})
        for k, v in kv.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        (self.state / "overrides.json").write_text(json.dumps(o, indent=1))
        return cur

    # ---------------------------------------------------------------- tags (sidecar + optional file write)
    def _tags_file(self) -> Path:
        return self.state / "tags.json"

    def user_tags(self, tid: str | None = None) -> dict:
        try:
            allt = json.loads(self._tags_file().read_text())
        except (OSError, ValueError):
            allt = {}
        return allt if tid is None else allt.get(tid, {})

    def set_user_tags(self, ref: str, values: dict, write_to_file: bool = False, backup: bool = True, clear: list[str] | None = None) -> dict:
        """Validate and store tags in .openmixing/tags.json; with write_to_file also write the container-capable ones into the file
        (the original is copied to .openmixing/backups/ first when `backup`). The track id stays the same after a file write."""
        t = self.resolve(ref)
        vals = {k: taglib.clean(k, v) for k, v in values.items() if v is not None}
        bad = [k for k in [*vals, *(clear or [])] if k not in (*taglib.FILE_FIELDS, *taglib.SIDECAR_ONLY)]
        if bad:
            raise OpenMixingError("invalid_value", f"unknown tag fields {bad}", bad[0],
                                f"use {list(taglib.FILE_FIELDS) + list(taglib.SIDECAR_ONLY)}")
        allt = self.user_tags()
        cur = allt.setdefault(t["id"], {})
        for k in clear or []:
            cur.pop(k, None)
        cur.update(vals)
        written: dict = {"written": [], "skipped": {}}
        backup_path = None
        file_vals = {k: v for k, v in vals.items() if k in taglib.FILE_FIELDS}
        if write_to_file and file_vals:
            if backup:
                dst = self.output_path(self.state / "backups" / t["id"] / Path(t["path"]).name, "backup")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():  # keep the first (original) copy
                    shutil.copy2(t["path"], dst)
                backup_path = str(dst)
            written = taglib.write_file(t["path"], file_vals)
            self._pin_after_write(t)
        self._tags_file().write_text(json.dumps(allt, indent=1))
        sk = {k: "sidecar only (not a file tag)" for k in vals if k in taglib.SIDECAR_ONLY}
        return {"track": t["file"], "tags": cur, "file_written": written["written"], "skipped": {**written["skipped"], **sk},
                "backup": backup_path, "mutated_file": bool(written["written"])}

    def _pin_after_write(self, t: dict) -> None:
        """A tag write changes the file bytes: keep the track id, and refresh the cached fingerprints and tags instead of re-analysing."""
        st = os.stat(t["path"])
        cache = self._hashes()
        cache[t["file"]] = {"key": f"{st.st_mtime_ns}:{st.st_size}", "id": t["id"]}
        self._hash_file().write_text(json.dumps(cache))
        fp = self._fingerprint(t["path"])
        try:
            d = json.loads(self._f(t["id"]).read_text())
            d.update(fingerprint=fp, tags=read_tags(t["path"]))
            self._f(t["id"]).write_text(json.dumps(d))
        except (OSError, ValueError):
            pass
        sm = self.state / "stems" / t["id"] / "stems.json"
        try:
            m = json.loads(sm.read_text())
            m["fingerprint"] = fp
            sm.write_text(json.dumps(m))
        except (OSError, ValueError):
            pass

    def merged_tags(self, tid: str, embedded: dict | None = None) -> dict:
        """File tags overlaid with the user's sidecar values."""
        return {**(embedded or {}), **{k: v for k, v in self.user_tags(tid).items()}}

    # ---------------------------------------------------------------- hot cues
    def cues(self, tid: str) -> dict:
        try:
            return json.loads((self.state / "cues.json").read_text()).get(tid, {})
        except (OSError, ValueError):
            return {}

    def _write_cues(self, tid: str, cues: dict) -> None:
        try:
            allc = json.loads((self.state / "cues.json").read_text())
        except (OSError, ValueError):
            allc = {}
        allc[tid] = cues
        (self.state / "cues.json").write_text(json.dumps(allc, indent=1))

    def set_cue(self, tid: str, name: str, bar: float, color: str | None = None, note: str = "") -> dict:
        c = self.cues(tid)
        c[name] = {"bar": bar, "color": color, "note": note}
        self._write_cues(tid, c)
        return c

    def delete_cue(self, tid: str, name: str) -> dict:
        c = self.cues(tid)
        if name not in c:
            raise KeyError(f"no cue {name!r} on this track; cues: {sorted(c)}")
        del c[name]
        self._write_cues(tid, c)
        return c

    def to_bars(self, ref: str, value, field: str = "position") -> float:
        """Resolve a flexible position (bars, beats, seconds, cue or section name) on a track to bars."""
        from .units import to_bars
        e = self.effective(ref)
        d = self.analyze(e["id"])
        cues = {k: v["bar"] for k, v in self.cues(e["id"]).items()}
        return to_bars(value, e, cues, d["structure"]["sections"], field)

    def out_bars(self, clip, value, field: str = "position") -> float:
        """Position inside `clip`'s playback in output bars; accepts bars, beats, cue / section names and source seconds."""
        from .units import out_bars
        return out_bars(value, clip.src_start_beat / 4.0, lambda v: self.to_bars(clip.track, v, field), field)

    # ---------------------------------------------------------------- online
    def online(self, ref: str, providers=None, artist=None, title=None, refresh=False) -> dict:
        t = self.resolve(ref)
        f = self._f(t["id"], ".online")
        if f.exists() and not refresh and not (providers or artist or title):
            return json.loads(f.read_text())
        d = self.analyze(t["id"])
        res = online.lookup(t["path"], Path(t["file"]).stem, d["tags"], providers, artist, title, d["tempo"]["bpm"])
        if not (providers or artist or title) or not f.exists():
            f.write_text(json.dumps(res))
        return res

    def cached_online(self, tid: str) -> dict | None:
        try:
            return json.loads(self._f(tid, ".online").read_text())
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------- merged views
    def effective(self, ref: str) -> dict:
        """Analysis with manual overrides applied: what the renderer and planner actually use."""
        t = self.resolve(ref)
        d = self.analyze(t["id"])
        ov = self.overrides().get(t["id"], {})
        tempo, key, loud = d["tempo"], d["key"], d["loudness"]
        eff = {"id": t["id"], "file": t["file"], "path": t["path"],
               "bpm": float(ov.get("bpm", tempo["bpm"])), "first_downbeat": float(ov.get("first_downbeat", tempo["first_downbeat_sec"])),
               "gain_db": float(ov.get("gain_db", loud["suggested_gain_db"])), "duration": float(d["duration_sec"]),
               "camelot": ov.get("camelot", key["camelot"]), "key": ov.get("key", key["key"]),
               "energy": d["energy"]["level_1_10"], "overrides": ov, "stems": self.stem_paths(t),
               "stems_legacy": self.stems_legacy(t)}
        eff["stem_quality"] = (self._stem_meta(t) or {}).get("quality") if eff["stems"] else None
        eff["bass_carrier"] = stemlib.bass_carrier(eff)
        return eff

    def render_info(self, tid: str) -> dict:
        return self.effective(tid)

    # ---------------------------------------------------------------- sets
    def set_path(self, name: str) -> Path:
        return self.output_path(self.state / "sets" / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".json"), "name")

    def load_set(self, name: str) -> SetModel:
        p = self.set_path(name)
        if not p.exists():
            raise KeyError(f"no set {name!r}; sets: {self.list_sets()}")
        return SetModel.load(p)

    def save_set(self, model: SetModel) -> None:
        model.save(self.set_path(model.name))

    def list_sets(self) -> list[str]:
        return sorted(p.stem for p in (self.state / "sets").glob("*.json"))
