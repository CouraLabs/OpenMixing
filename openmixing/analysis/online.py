"""Online track metadata. Free/no-key: MusicBrainz, Deezer, iTunes, ReccoBeats. Key-gated: Last.fm
(LASTFM_API_KEY), Discogs (DISCOGS_TOKEN), AcoustID (ACOUSTID_API_KEY + the `fpcalc` binary).

Every provider returns a dict or {"error": ...}; none can raise, so one dead API never blocks the rest.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from urllib.parse import quote

import requests

UA = "OpenMixing/0.1 (https://github.com/couralabs/openmixing)"
TIMEOUT = 12
_mb_lock, _mb_last = threading.Lock(), [0.0]

PROVIDERS = ["musicbrainz", "deezer", "itunes", "reccobeats", "lastfm", "discogs", "acoustid"]


def _get(url, params=None, headers=None):
    r = requests.get(url, params=params, headers={"User-Agent": UA, **(headers or {})}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def guess_artist_title(name: str, tags: dict) -> tuple[str | None, str]:
    artist, title = tags.get("artist"), tags.get("title")
    if artist and title and title != name:
        return str(artist), str(title)
    m = re.match(r"^\s*(?:\d+\s*[-.]\s*)?(.+?)\s+-\s+(.+?)\s*$", name)
    if m:
        return m.group(1), re.sub(r"\s*[\[(](?:official|original mix|extended mix|free download)[^\])]*[\])]", "", m.group(2), flags=re.I)
    return (str(artist) if artist else None), str(title or name)


def musicbrainz(artist, title, isrc=None):
    try:
        with _mb_lock:  # MusicBrainz allows 1 request per second
            wait = 1.1 - (time.time() - _mb_last[0])
            if wait > 0:
                time.sleep(wait)
            q = f'isrc:{isrc}' if isrc else f'recording:"{title}"' + (f' AND artist:"{artist}"' if artist else "")
            res = _get("https://musicbrainz.org/ws/2/recording", {"query": q, "fmt": "json", "limit": 3})
            _mb_last[0] = time.time()
            recs = [r for r in res.get("recordings", []) if r.get("score", 0) >= 80]
            if not recs:
                return {"error": "no match"}
            time.sleep(1.1)
            r = _get(f"https://musicbrainz.org/ws/2/recording/{recs[0]['id']}",
                     {"inc": "genres+tags+isrcs+releases+artist-credits", "fmt": "json"})
            _mb_last[0] = time.time()
        rel = (r.get("releases") or [{}])[0]
        return {
            "id": r["id"], "title": r.get("title"), "artist": " ".join(f"{a['name']}{a.get('joinphrase', '')}" for a in r.get("artist-credit", [])),
            "length_ms": r.get("length"), "isrcs": r.get("isrcs", []), "first_release_date": r.get("first-release-date"),
            "release": rel.get("title"), "release_date": rel.get("date"), "country": rel.get("country"),
            "genres": [g["name"] for g in sorted(r.get("genres", []), key=lambda g: -g.get("count", 0))],
            "tags": [t["name"] for t in sorted(r.get("tags", []), key=lambda t: -t.get("count", 0))][:15],
            "match_score": recs[0]["score"],
        }
    except Exception as e:
        return {"error": str(e)}


def deezer(artist, title):
    try:
        q = f"{artist} {title}" if artist else title
        hits = _get("https://api.deezer.com/search", {"q": q, "limit": 3}).get("data", [])
        if not hits:
            return {"error": "no match"}
        t = _get(f"https://api.deezer.com/track/{hits[0]['id']}")
        out = {"id": t["id"], "title": t.get("title"), "artist": t.get("artist", {}).get("name"),
               "bpm": t.get("bpm") or None, "gain_db": t.get("gain"), "isrc": t.get("isrc"),
               "release_date": t.get("release_date"), "rank": t.get("rank"), "explicit": t.get("explicit_lyrics"),
               "duration_sec": t.get("duration"), "album": t.get("album", {}).get("title"), "link": t.get("link"),
               "preview_url": t.get("preview")}
        if t.get("album", {}).get("id"):
            try:
                alb = _get(f"https://api.deezer.com/album/{t['album']['id']}")
                out["genres"] = [g["name"] for g in alb.get("genres", {}).get("data", [])]
                out["label"] = alb.get("label")
            except Exception:
                pass
        return out
    except Exception as e:
        return {"error": str(e)}


def itunes(artist, title):
    try:
        res = _get("https://itunes.apple.com/search", {"term": f"{artist or ''} {title}".strip(), "entity": "song", "limit": 3})
        if not res.get("results"):
            return {"error": "no match"}
        t = res["results"][0]
        return {"title": t.get("trackName"), "artist": t.get("artistName"), "album": t.get("collectionName"),
                "genre": t.get("primaryGenreName"), "release_date": t.get("releaseDate"), "duration_sec": (t.get("trackTimeMillis") or 0) / 1000 or None,
                "explicit": t.get("trackExplicitness"), "artwork_url": (t.get("artworkUrl100") or "").replace("100x100", "600x600"),
                "link": t.get("trackViewUrl"), "preview_url": t.get("previewUrl")}
    except Exception as e:
        return {"error": str(e)}


def reccobeats(isrc):
    """Spotify-style audio features (acousticness, danceability, energy, valence, tempo ...) by ISRC."""
    if not isrc:
        return {"error": "needs an ISRC (comes from MusicBrainz / Deezer / the file tags)"}
    try:
        c = _get("https://api.reccobeats.com/v1/audio-features", {"ids": isrc}).get("content", [])
        if not c:
            return {"error": "no match"}
        f = c[0]
        keep = ("acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "key", "mode")
        out = {k: f[k] for k in keep if k in f}
        out["spotify_url"] = f.get("href")
        return out
    except Exception as e:
        return {"error": str(e)}


def lastfm(artist, title):
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        return {"error": "set LASTFM_API_KEY to enable"}
    try:
        base = {"api_key": key, "format": "json", "artist": artist or "", "track": title, "autocorrect": 1}
        info = _get("https://ws.audioscrobbler.com/2.0/", {**base, "method": "track.getInfo"}).get("track", {})
        tags = _get("https://ws.audioscrobbler.com/2.0/", {**base, "method": "track.getTopTags"}).get("toptags", {}).get("tag", [])
        return {"listeners": int(info.get("listeners", 0)), "playcount": int(info.get("playcount", 0)),
                "tags": [t["name"] for t in tags[:15]], "url": info.get("url")}
    except Exception as e:
        return {"error": str(e)}


def discogs(artist, title):
    tok = os.environ.get("DISCOGS_TOKEN")
    if not tok:
        return {"error": "set DISCOGS_TOKEN to enable"}
    try:
        res = _get("https://api.discogs.com/database/search", {"artist": artist, "track": title, "type": "release", "per_page": 3},
                   {"Authorization": f"Discogs token={tok}"}).get("results", [])
        if not res:
            return {"error": "no match"}
        r = res[0]
        return {"title": r.get("title"), "year": r.get("year"), "genres": r.get("genre", []), "styles": r.get("style", []),
                "labels": r.get("label", [])[:3], "country": r.get("country"), "url": "https://www.discogs.com" + r.get("uri", "")}
    except Exception as e:
        return {"error": str(e)}


def acoustid(path):
    key = os.environ.get("ACOUSTID_API_KEY")
    if not key:
        return {"error": "set ACOUSTID_API_KEY to enable"}
    if not shutil.which("fpcalc"):
        return {"error": "needs the chromaprint `fpcalc` binary (brew install chromaprint)"}
    try:
        import subprocess

        o = subprocess.run(["fpcalc", "-json", path], capture_output=True, text=True, timeout=60)
        import json

        fp = json.loads(o.stdout)
        res = requests.post("https://api.acoustid.org/v2/lookup", data={"client": key, "duration": int(fp["duration"]),
                            "fingerprint": fp["fingerprint"], "meta": "recordings releasegroups"}, timeout=TIMEOUT).json()
        best = max(res.get("results", []), key=lambda r: r.get("score", 0), default=None)
        if not best or not best.get("recordings"):
            return {"error": "no match"}
        rec = best["recordings"][0]
        return {"score": best["score"], "musicbrainz_recording_id": rec.get("id"), "title": rec.get("title"),
                "artists": [a["name"] for a in rec.get("artists", [])]}
    except Exception as e:
        return {"error": str(e)}


def lookup(path: str, name: str, tags: dict, providers: list[str] | None = None, artist: str | None = None,
           title: str | None = None, analysis_bpm: float | None = None) -> dict:
    a, t = guess_artist_title(name, tags)
    artist, title = artist or a, title or t
    want = providers or PROVIDERS
    res: dict = {"query": {"artist": artist, "title": title}}
    isrc = tags.get("isrc")
    if "deezer" in want:
        res["deezer"] = deezer(artist, title)
        isrc = isrc or res["deezer"].get("isrc")
    if "musicbrainz" in want:
        res["musicbrainz"] = musicbrainz(artist, title, tags.get("isrc"))
        isrc = isrc or next(iter(res["musicbrainz"].get("isrcs") or []), None)
    if "itunes" in want:
        res["itunes"] = itunes(artist, title)
    if "reccobeats" in want:
        res["reccobeats"] = reccobeats(isrc)
    if "lastfm" in want:
        res["lastfm"] = lastfm(artist, title)
    if "discogs" in want:
        res["discogs"] = discogs(artist, title)
    if "acoustid" in want:
        res["acoustid"] = acoustid(path)
    ok = lambda p: p in res and "error" not in res[p]
    genres = []
    for p, k in (("musicbrainz", "genres"), ("deezer", "genres"), ("discogs", "genres"), ("discogs", "styles"), ("lastfm", "tags")):
        if ok(p):
            genres += res[p].get(k) or []
    if ok("itunes") and res["itunes"].get("genre"):
        genres.append(res["itunes"]["genre"])
    seen, uniq = set(), []
    for g in genres:
        if g.lower() not in seen:
            seen.add(g.lower())
            uniq.append(g)
    bpms = {p: v for p, v in (("deezer", res.get("deezer", {}).get("bpm")), ("reccobeats", res.get("reccobeats", {}).get("tempo"))) if v}
    res["consensus"] = {
        "isrc": isrc, "genres": uniq[:12], "reference_bpm": bpms,
        "release_date": min((d[:10] for p, k in (("musicbrainz", "first_release_date"), ("deezer", "release_date"), ("itunes", "release_date"))
                             if ok(p) and (d := res[p].get(k))), default=None),
        "bpm_check": None if not (bpms and analysis_bpm) else {
            p: ("agrees" if min(abs(v - analysis_bpm), abs(v / 2 - analysis_bpm), abs(v * 2 - analysis_bpm)) < 2 else f"differs ({v})")
            for p, v in bpms.items()},
    }
    return res
