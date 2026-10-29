from __future__ import annotations

import threading
import time

from .http import Miss, first, get, provider

_mb_lock, _mb_last = threading.Lock(), [0.0]


def _mb_get(url, params):
    with _mb_lock:
        wait = 1.1 - (time.time() - _mb_last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            return get(url, params)
        finally:
            _mb_last[0] = time.time()


@provider
def musicbrainz(artist, title, isrc=None):
    q = f'isrc:{isrc}' if isrc else f'recording:"{title}"' + (f' AND artist:"{artist}"' if artist else "")
    res = _mb_get("https://musicbrainz.org/ws/2/recording", {"query": q, "fmt": "json", "limit": 3})
    rec = first([r for r in res.get("recordings", []) if r.get("score", 0) >= 80])
    r = _mb_get(f"https://musicbrainz.org/ws/2/recording/{rec['id']}", {"inc": "genres+tags+isrcs+releases+artist-credits", "fmt": "json"})
    rel = (r.get("releases") or [{}])[0]
    by_count = lambda xs: [x["name"] for x in sorted(xs, key=lambda x: -x.get("count", 0))]
    return {
        "id": r["id"], "title": r.get("title"), "artist": " ".join(f"{a['name']}{a.get('joinphrase', '')}" for a in r.get("artist-credit", [])),
        "length_ms": r.get("length"), "isrcs": r.get("isrcs", []), "first_release_date": r.get("first-release-date"),
        "release": rel.get("title"), "release_date": rel.get("date"), "country": rel.get("country"),
        "genres": by_count(r.get("genres", [])), "tags": by_count(r.get("tags", []))[:15], "match_score": rec["score"],
    }


def _deezer_album(album_id) -> dict:
    try:
        alb = get(f"https://api.deezer.com/album/{album_id}")
        return {"genres": [g["name"] for g in alb.get("genres", {}).get("data", [])], "label": alb.get("label")}
    except Exception:
        return {}


@provider
def deezer(artist, title):
    hit = first(get("https://api.deezer.com/search", {"q": f"{artist} {title}" if artist else title, "limit": 3}).get("data", []))
    t = get(f"https://api.deezer.com/track/{hit['id']}")
    album = t.get("album", {})
    out = {"id": t["id"], "title": t.get("title"), "artist": t.get("artist", {}).get("name"),
           "bpm": t.get("bpm") or None, "gain_db": t.get("gain"), "isrc": t.get("isrc"),
           "release_date": t.get("release_date"), "rank": t.get("rank"), "explicit": t.get("explicit_lyrics"),
           "duration_sec": t.get("duration"), "album": album.get("title"), "link": t.get("link"), "preview_url": t.get("preview")}
    return {**out, **(_deezer_album(album["id"]) if album.get("id") else {})}


@provider
def itunes(artist, title):
    t = first(get("https://itunes.apple.com/search", {"term": f"{artist or ''} {title}".strip(), "entity": "song", "limit": 3}).get("results"))
    return {"title": t.get("trackName"), "artist": t.get("artistName"), "album": t.get("collectionName"),
            "genre": t.get("primaryGenreName"), "release_date": t.get("releaseDate"),
            "duration_sec": (t.get("trackTimeMillis") or 0) / 1000 or None,
            "explicit": t.get("trackExplicitness"), "artwork_url": (t.get("artworkUrl100") or "").replace("100x100", "600x600"),
            "link": t.get("trackViewUrl"), "preview_url": t.get("previewUrl")}


RECCO_KEYS = ("acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence",
              "key", "mode")


@provider
def reccobeats(isrc):
    if not isrc:
        raise Miss("needs an ISRC (comes from MusicBrainz / Deezer / the file tags)")
    f = first(get("https://api.reccobeats.com/v1/audio-features", {"ids": isrc}).get("content", []))
    return {**{k: f[k] for k in RECCO_KEYS if k in f}, "spotify_url": f.get("href")}
