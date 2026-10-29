from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from pathlib import Path

from ..analysis.analyze import AUDIO_EXT
from ..core.errors import OpenMixingError
from ..core.jsonio import read_json, write_json
from ..model.setmodel import SetModel

STALE_SEC = 3600


class Catalog:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("OPENMIXING_DIR") or Path.cwd()).expanduser().resolve()
        self.state = self.root / ".openmixing"
        self.scan_lock = threading.RLock()
        for d in ("analysis", "sets", "renders", "stems", "samples"):
            (self.state / d).mkdir(parents=True, exist_ok=True)
        self._sweep()

    def _sweep(self) -> None:
        old = time.time() - STALE_SEC
        for pat in ("*.raw.wav", "*.part", "*.enc.part"):
            for p in (self.state / "renders").glob(pat):
                if p.stat().st_mtime < old:
                    p.unlink(missing_ok=True)
        for p in self.state.rglob(".*.tmp"):
            if p.stat().st_mtime < old:
                p.unlink(missing_ok=True)

    def scan(self) -> list[dict]:
        with self.scan_lock:
            return self._scan()

    def _scan(self) -> list[dict]:
        out, seen, cache, files = [], {}, self._hashes(), set()
        dirty = False
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".") and ".openmixing" not in p.parts and self._inside(p, self.root):
                rel = str(p.relative_to(self.root))
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.add(rel)
                key = f"{st.st_mtime_ns}:{st.st_size}"
                hit = cache.get(rel)
                if not hit or hit["key"] != key:
                    hit = cache[rel] = {"key": key, "id": self.content_id(p)}
                    dirty = True
                tid = hit["id"]
                if tid in seen:
                    tid = f"{tid}-{hashlib.blake2b(rel.encode(), digest_size=2).hexdigest()}"
                    out.append({"id": tid, "file": rel, "path": str(p), "duplicate_of": seen[hit["id"]]})
                    continue
                seen[tid] = rel
                out.append({"id": tid, "file": rel, "path": str(p)})
        if stale := set(cache) - files:
            cache = {k: v for k, v in cache.items() if k not in stale}
            dirty = True
        if dirty:
            write_json(self._hash_file(), cache)
        return out

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
        return self._allowed(path, self.root, field, "inputs", "the tracks folder")

    def output_path(self, path: Path | str, field: str = "path") -> Path:
        return self._allowed(path, self.state.resolve(), field, "outputs", "the .openmixing/ state folder")

    @staticmethod
    def safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).lstrip(".") or "unnamed"

    def render_path(self, name: str, fmt: str) -> Path:
        return self.output_path(self.state / "renders" / f"{self.safe_name(name)}.{fmt}", "name")

    def _hash_file(self) -> Path:
        return self.state / "hashes.json"

    def _hashes(self) -> dict:
        return read_json(self._hash_file(), {})

    @staticmethod
    def content_id(path: Path | str) -> str:
        h = hashlib.blake2b(digest_size=6)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def resolve(self, ref: str) -> dict:
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
