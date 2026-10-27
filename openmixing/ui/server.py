"""Local player for the rendered set: static page + JSON state + audio with HTTP Range support."""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..errors import OpenMixingError
from ..render import timeline
from ..setmodel import SetModel, TempoMap, resolve_starts

WEB = Path(__file__).parent / "index.html"
AUDIO_EXT = (".flac", ".wav", ".mp3")


def build_state(lib, model: SetModel, meta: dict) -> dict:
    """Everything the player needs, in set-time seconds."""
    info = lib.render_info
    starts = resolve_starts(model, lambda t: info(t)["bpm"])
    clips = []
    for c in model.clips:
        e = info(c.track)
        d = lib.analyze(c.track)
        tm = TempoMap(c, e["bpm"])
        t0 = starts[c.id]
        sec = lambda b: round(t0 + float(tm.beat_to_sec(max(min(b, c.length_beats), 0))), 3)
        beat_sec = lambda b: round(t0 + float(tm.beat_to_sec(b)), 3)
        clips.append({
            "id": c.id, "label": c.label, "file": e["file"], "title": d["tags"].get("title"), "artist": d["tags"].get("artist"),
            "native_bpm": round(e["bpm"], 2), "camelot": e["camelot"], "key": e["key"], "energy": e["energy"],
            "start": sec(0), "end": sec(c.length_beats), "src_start_beat": c.src_start_beat, "length_beats": c.length_beats,
            "tempo": [[beat_sec(b), v] for b, v in (c.tempo.points or [[0, e["bpm"]]])],
            "lanes": {k: [[beat_sec(b), v] for b, v in l.points] for k, l in c.lanes.items() if l.points},
            "fx": [{"id": f.id, "fx": f.fx, "start": beat_sec(f.beat), "end": sec(f.beat + f.duration_beats), "beats": f.beats, "wet": f.wet} for f in c.fx],
            "loops": [{"id": l.id, "start": beat_sec(l.beat), "end": sec(l.beat + l.length_beats), "size_beats": l.size_beats, "roll": l.roll} for l in c.loops],
            "section_map": [{"label": s["label"], "start": s["start_sec"], "end": s["end_sec"]} for s in d["structure"]["sections"]],
        })
    keep = {k: meta.get(k) for k in ("start_sec", "end_sec", "duration_sec", "integrated_lufs", "loudness_range_lu", "true_peak_dbtp", "peaks", "peak_block_sec", "format")}
    return {"set": model.name, "version": model.version, "audio": Path(meta["path"]).name, "clips": clips, **keep}


def start(lib, port: int = 8765) -> dict:
    renders = lib.state / "renders"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep stdout clean (MCP stdio)
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        do_HEAD = lambda self: self.do_GET()

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/sets":
                sides = sorted((p for p in renders.glob("*.json") if ".window" not in p.name and not p.name.startswith("preview_")),
                               key=lambda p: -p.stat().st_mtime)
                return self._send(200, json.dumps([p.stem for p in sides]).encode())
            if path == "/api/state":
                name = (urlparse(self.path).query or "").removeprefix("set=")
                sides = sorted((p for p in renders.glob("*.json") if ".window" not in p.name and not p.name.startswith("preview_")),
                               key=lambda p: -p.stat().st_mtime)
                side = next((p for p in sides if p.stem == name), sides[0] if sides else None)
                if not side:
                    return self._send(404, b'{"error":"nothing rendered yet: call the render tool"}')
                return self._send(200, side.read_bytes())
            if path.startswith("/audio/"):
                try:
                    f = lib.output_path(renders / Path(path[7:]).name, "path")
                except OpenMixingError:
                    return self._send(404, b"{}")
                if not f.is_file() or f.suffix not in AUDIO_EXT:
                    return self._send(404, b"{}")
                size, rng = f.stat().st_size, self.headers.get("Range")
                a, b = 0, size - 1
                if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
                    if m.group(1):
                        a = int(m.group(1))
                        b = int(m.group(2)) if m.group(2) else size - 1
                    elif m.group(2):
                        a = max(size - int(m.group(2)), 0)
                b = min(b, size - 1)
                with open(f, "rb") as fh:
                    fh.seek(a)
                    data = fh.read(b - a + 1)
                ctype = mimetypes.guess_type(f.name)[0] or "audio/flac"
                return self._send(206 if rng else 200, data, ctype, {"Accept-Ranges": "bytes", "Content-Range": f"bytes {a}-{b}/{size}"})
            self._send(404, b"{}")

    for p in range(port, port + 30):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            break
        except OSError:
            continue
    else:
        raise OSError("no free port for the player")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return {"url": f"http://127.0.0.1:{p}/", "server": srv}
