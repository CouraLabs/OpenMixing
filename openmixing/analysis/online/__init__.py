from __future__ import annotations

from .catalogs import deezer, itunes, musicbrainz, reccobeats
from .consensus import consensus
from .http import guess_artist_title
from .keyed import acoustid, discogs, lastfm

PROVIDERS = ["musicbrainz", "deezer", "itunes", "reccobeats", "lastfm", "discogs", "acoustid"]


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
    calls = {"itunes": lambda: itunes(artist, title), "reccobeats": lambda: reccobeats(isrc), "lastfm": lambda: lastfm(artist, title),
             "discogs": lambda: discogs(artist, title), "acoustid": lambda: acoustid(path)}
    res.update({p: call() for p, call in calls.items() if p in want})
    res["consensus"] = consensus(res, isrc, analysis_bpm)
    return res
