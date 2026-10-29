from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..analysis.meta import read_tags
from ..core.errors import OpenMixingError
from ..core.jsonio import read_json, write_json
from . import tags as taglib


class Annotations:
    def overrides(self) -> dict:
        return read_json(self.state / "overrides.json", {}, strict=True)

    def set_override(self, tid: str, **kv) -> dict:
        o = self.overrides()
        cur = o.setdefault(tid, {})
        for k, v in kv.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        write_json(self.state / "overrides.json", o, 1)
        return cur

    def _tags_file(self) -> Path:
        return self.state / "tags.json"

    def user_tags(self, tid: str | None = None) -> dict:
        allt = read_json(self._tags_file(), {}, strict=True)
        return allt if tid is None else allt.get(tid, {})

    def set_user_tags(self, ref: str, values: dict, write_to_file: bool = False, backup: bool = True, clear: list[str] | None = None) -> dict:
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
                if not dst.exists():
                    shutil.copy2(t["path"], dst)
                backup_path = str(dst)
            written = taglib.write_file(t["path"], file_vals)
            self._pin_after_write(t)
        write_json(self._tags_file(), allt, 1)
        sk = {k: "sidecar only (not a file tag)" for k in vals if k in taglib.SIDECAR_ONLY}
        return {"track": t["file"], "tags": cur, "file_written": written["written"], "skipped": {**written["skipped"], **sk},
                "backup": backup_path, "mutated_file": bool(written["written"])}

    def _pin_after_write(self, t: dict) -> None:
        st = os.stat(t["path"])
        cache = self._hashes()
        cache[t["file"]] = {"key": f"{st.st_mtime_ns}:{st.st_size}", "id": t["id"]}
        write_json(self._hash_file(), cache)
        fp = self._fingerprint(t["path"])
        if (d := read_json(self._f(t["id"]))) is not None:
            d.update(fingerprint=fp, tags=read_tags(t["path"]))
            write_json(self._f(t["id"]), d)
        sm = self.state / "stems" / t["id"] / "stems.json"
        if (m := read_json(sm)) is not None:
            m["fingerprint"] = fp
            write_json(sm, m)

    def merged_tags(self, tid: str, embedded: dict | None = None) -> dict:
        return {**(embedded or {}), **{k: v for k, v in self.user_tags(tid).items()}}

    def cues(self, tid: str) -> dict:
        return read_json(self.state / "cues.json", {}, strict=True).get(tid, {})

    def _write_cues(self, tid: str, cues: dict) -> None:
        allc = read_json(self.state / "cues.json", {}, strict=True)
        allc[tid] = cues
        write_json(self.state / "cues.json", allc, 1)

    def set_cue(self, tid: str, name: str, bar: float, color: str | None = None, note: str = "") -> dict:
        c = self.cues(tid)
        c[name] = {"bar": bar, "color": color, "note": note}
        self._write_cues(tid, c)
        return c

    def delete_cue(self, tid: str, name: str) -> dict:
        c = self.cues(tid)
        if name not in c:
            raise OpenMixingError("not_found", f"no cue {name!r} on this track", "name", f"cues: {sorted(c)}")
        del c[name]
        self._write_cues(tid, c)
        return c
