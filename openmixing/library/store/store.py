from __future__ import annotations

from pathlib import Path

from ...core.errors import OpenMixingError
from ...core.jsonio import read_json, write_json
from . import rules as ruleslib
from .history import History

SORTS = ("name", "bpm", "energy", "rating", "duration", "play_count", "last_played")


class Store:
    def __init__(self, state: Path):
        self.cfile, self.rfile = state / "collections.json", state / "crates.json"
        self.history = History(state / "history.jsonl")

    def collections(self) -> dict:
        return read_json(self.cfile, {}, strict=True)

    def collection(self, name: str) -> dict:
        c = self.collections()
        if name not in c:
            raise OpenMixingError("not_found", f"no collection {name!r}", "name", f"collections: {sorted(c)}")
        return c[name]

    def create(self, name: str, notes: str = "") -> dict:
        c = self.collections()
        cur = c.setdefault(name, {"tracks": [], "notes": ""})
        cur["notes"] = notes or cur["notes"]
        write_json(self.cfile, c, 1)
        return c[name]

    def change(self, name: str, add: list[str], remove: list[str]) -> dict:
        c = self.collections()
        self.collection(name)
        cur = [t for t in c[name]["tracks"] if t not in remove]
        c[name]["tracks"] = cur + [t for t in dict.fromkeys(add) if t not in cur]
        write_json(self.cfile, c, 1)
        return c[name]

    def crates(self) -> dict:
        return read_json(self.rfile, {}, strict=True)

    def crate(self, name: str) -> dict:
        c = self.crates()
        if name not in c:
            raise OpenMixingError("not_found", f"no crate {name!r}", "crate", f"crates: {sorted(c)}")
        return c[name]

    def save_crate(self, name: str, rules: dict, sort: str = "name") -> dict:
        ruleslib.check(rules)
        if sort not in SORTS:
            raise OpenMixingError("invalid_value", f"unknown sort {sort!r}", "sort", f"use one of {list(SORTS)}")
        c = self.crates()
        c[name] = {"rules": rules, "sort": sort}
        write_json(self.rfile, c, 1)
        return c[name]
