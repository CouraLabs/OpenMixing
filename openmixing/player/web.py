from __future__ import annotations

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..core.errors import OpenMixingError

WEB = (Path(__file__).parent / "static").resolve()
AUDIO_EXT = (".flac", ".wav", ".mp3")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}


def static_file(path: str) -> Path | None:
    f = (WEB / path.lstrip("/")).resolve()
    return f if f.is_file() and f.is_relative_to(WEB) and f.suffix in STATIC_TYPES else None


def rendered(renders: Path) -> list[Path]:
    return sorted((p for p in renders.glob("*.json") if ".window" not in p.name and not p.name.startswith("preview_")),
                  key=lambda p: -p.stat().st_mtime)


def start(lib, port: int = 8765) -> dict:
    renders = lib.state / "renders"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
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
            if path == "/" or path.startswith("/static/"):
                f = static_file("index.html" if path == "/" else path.removeprefix("/static/"))
                return self._send(200, f.read_bytes(), STATIC_TYPES[f.suffix]) if f else self._send(404, b"{}")
            if path == "/api/sets":
                return self._send(200, json.dumps([p.stem for p in rendered(renders)]).encode())
            if path in ("/api/state", "/api/version"):
                name = unquote((urlparse(self.path).query or "").removeprefix("set="))
                sides = rendered(renders)
                side = next((p for p in sides if p.stem == name), sides[0] if sides else None)
                if not side:
                    return self._send(404, b'{"error":"nothing rendered yet: call the render tool"}')
                if path == "/api/version":
                    st = side.stat()
                    return self._send(200, json.dumps({"stamp": f"{side.stem}:{st.st_mtime_ns}:{st.st_size}"}).encode())
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
                ctype = mimetypes.guess_type(f.name)[0] or "audio/mpeg"
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
