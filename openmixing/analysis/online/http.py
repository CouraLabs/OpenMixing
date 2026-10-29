from __future__ import annotations

import functools
import os
import re

import requests

UA = "OpenMixing/0.1 (https://github.com/couralabs/openmixing)"
TIMEOUT = 12


class Miss(Exception):
    pass


def get(url, params=None, headers=None):
    r = requests.get(url, params=params, headers={"User-Agent": UA, **(headers or {})}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def env_key(name: str) -> str:
    if not (v := os.environ.get(name)):
        raise Miss(f"set {name} to enable")
    return v


def first(items: list, what: str = "no match"):
    if not items:
        raise Miss(what)
    return items[0]


def provider(fn):
    @functools.wraps(fn)
    def run(*args):
        try:
            return fn(*args)
        except Exception as e:
            return {"error": str(e)}
    return run


def guess_artist_title(name: str, tags: dict) -> tuple[str | None, str]:
    artist, title = tags.get("artist"), tags.get("title")
    if artist and title and title != name:
        return str(artist), str(title)
    m = re.match(r"^\s*(?:\d+\s*[-.]\s*)?(.+?)\s+-\s+(.+?)\s*$", name)
    if m:
        return m.group(1), re.sub(r"\s*[\[(](?:official|original mix|extended mix|free download)[^\])]*[\])]", "", m.group(2), flags=re.I)
    return (str(artist) if artist else None), str(title or name)
