from __future__ import annotations

import json
import shutil
import subprocess

import requests

from .http import TIMEOUT, Miss, env_key, first, get, provider


@provider
def lastfm(artist, title):
    base = {"api_key": env_key("LASTFM_API_KEY"), "format": "json", "artist": artist or "", "track": title, "autocorrect": 1}
    info = get("https://ws.audioscrobbler.com/2.0/", {**base, "method": "track.getInfo"}).get("track", {})
    tags = get("https://ws.audioscrobbler.com/2.0/", {**base, "method": "track.getTopTags"}).get("toptags", {}).get("tag", [])
    return {"listeners": int(info.get("listeners", 0)), "playcount": int(info.get("playcount", 0)),
            "tags": [t["name"] for t in tags[:15]], "url": info.get("url")}


@provider
def discogs(artist, title):
    tok = env_key("DISCOGS_TOKEN")
    r = first(get("https://api.discogs.com/database/search", {"artist": artist, "track": title, "type": "release", "per_page": 3},
                  {"Authorization": f"Discogs token={tok}"}).get("results", []))
    return {"title": r.get("title"), "year": r.get("year"), "genres": r.get("genre", []), "styles": r.get("style", []),
            "labels": r.get("label", [])[:3], "country": r.get("country"), "url": "https://www.discogs.com" + r.get("uri", "")}


@provider
def acoustid(path):
    key = env_key("ACOUSTID_API_KEY")
    if not shutil.which("fpcalc"):
        raise Miss("needs the chromaprint `fpcalc` binary (brew install chromaprint)")
    fp = json.loads(subprocess.run(["fpcalc", "-json", path], capture_output=True, text=True, timeout=60).stdout)
    res = requests.post("https://api.acoustid.org/v2/lookup", data={"client": key, "duration": int(fp["duration"]),
                        "fingerprint": fp["fingerprint"], "meta": "recordings releasegroups"}, timeout=TIMEOUT).json()
    best = max(res.get("results", []), key=lambda r: r.get("score", 0), default=None)
    rec = first((best or {}).get("recordings"))
    return {"score": best["score"], "musicbrainz_recording_id": rec.get("id"), "title": rec.get("title"),
            "artists": [a["name"] for a in rec.get("artists", [])]}
