from __future__ import annotations

import json
import time
from pathlib import Path

TAGS = ("played", "rendered")


class History:
    def __init__(self, path: Path):
        self.path = path

    def append(self, set_name: str, rows: list[dict], tag: str = "played", render: str | None = None, ts: float | None = None) -> dict:
        rec = {"ts": time.time() if ts is None else ts, "set": set_name, "render": render, "tracks": rows, "tag": tag}
        lead = "\n" if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n") else ""
        with open(self.path, "a") as f:
            f.write(lead + json.dumps(rec) + "\n")
        return rec

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def plays(self, track: str | None = None, since_days: float | None = None, now: float | None = None) -> list[dict]:
        floor = (time.time() if now is None else now) - since_days * 86400.0 if since_days is not None else None
        return [{"ts": r["ts"], "set": r["set"], "render": r.get("render"), "tag": r["tag"], **t}
                for r in self.records() if r["tag"] == "played" and (floor is None or r["ts"] >= floor)
                for t in r["tracks"] if track is None or t["id"] == track]

    def stats(self) -> dict:
        out: dict = {}
        for p in self.plays():
            s = out.setdefault(p["id"], {"play_count": 0, "last_played": 0.0})
            s["play_count"] += 1
            s["last_played"] = max(s["last_played"], p["ts"])
        return out
