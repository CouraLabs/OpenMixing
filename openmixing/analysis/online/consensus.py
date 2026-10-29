from __future__ import annotations

GENRE_FIELDS = (("musicbrainz", "genres"), ("deezer", "genres"), ("discogs", "genres"), ("discogs", "styles"), ("lastfm", "tags"))
DATE_FIELDS = (("musicbrainz", "first_release_date"), ("deezer", "release_date"), ("itunes", "release_date"))


def _genres(res: dict, ok) -> list[str]:
    genres = [g for p, k in GENRE_FIELDS if ok(p) for g in res[p].get(k) or []]
    if ok("itunes") and res["itunes"].get("genre"):
        genres.append(res["itunes"]["genre"])
    seen: dict[str, str] = {}
    for g in genres:
        seen.setdefault(g.lower(), g)
    return list(seen.values())[:12]


def _bpm_check(bpms: dict, analysis_bpm: float | None) -> dict | None:
    if not (bpms and analysis_bpm):
        return None
    near = lambda v: min(abs(v - analysis_bpm), abs(v / 2 - analysis_bpm), abs(v * 2 - analysis_bpm)) < 2
    return {p: "agrees" if near(v) else f"differs ({v})" for p, v in bpms.items()}


def consensus(res: dict, isrc: str | None, analysis_bpm: float | None) -> dict:
    ok = lambda p: p in res and "error" not in res[p]
    bpms = {p: v for p, v in (("deezer", res.get("deezer", {}).get("bpm")), ("reccobeats", res.get("reccobeats", {}).get("tempo"))) if v}
    return {"isrc": isrc, "genres": _genres(res, ok), "reference_bpm": bpms,
            "release_date": min((d[:10] for p, k in DATE_FIELDS if ok(p) and (d := res[p].get(k))), default=None),
            "bpm_check": _bpm_check(bpms, analysis_bpm)}
