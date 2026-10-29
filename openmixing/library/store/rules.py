from __future__ import annotations

import time

from ...analysis.key import camelot_relation
from ...core.errors import OpenMixingError

NUM = ("eq", "ne", "between", "in", "gte", "lte")
TEXT = ("eq", "ne", "in", "contains")
FIELDS = {"bpm": (*NUM, "compatible_with"), "energy": NUM, "duration": NUM, "rating": NUM, "play_count": NUM,
          "camelot": (*TEXT, "compatible_with"), "color": TEXT, "artist": TEXT, "title": TEXT, "genre": TEXT, "notes": TEXT,
          "tag": ("contains", "in", "eq"), "has_stems": ("eq", "ne"), "last_played": ("within_days", "gte", "lte")}
DAY = 86400.0


def bad(path: str, message: str, fix: str) -> OpenMixingError:
    return OpenMixingError("invalid_value", message, path, fix)


def check(rules, path: str = "rules") -> None:
    if not isinstance(rules, dict) or not rules:
        raise bad(path, "a rule group is an object with 'all' or 'any'", 'use {"all": [rule, ...]} or {"any": [rule, ...]}')
    for k in rules:
        if k in ("all", "any"):
            if not isinstance(rules[k], list) or not rules[k]:
                raise bad(f"{path}.{k}", f"'{k}' must be a non-empty list", "list rules or nested groups")
            for i, r in enumerate(rules[k]):
                check(r, f"{path}.{k}[{i}]")
        elif k not in ("field", "op", "value"):
            raise bad(f"{path}.{k}", f"unknown key {k!r}", "use all / any, or field + op + value")
    if "all" in rules or "any" in rules:
        if len([k for k in rules if k in ("all", "any")]) != len(rules):
            raise bad(path, "a group holds only 'all' or 'any'", "nest rules inside the list")
        return
    f, op = rules.get("field"), rules.get("op")
    if f not in FIELDS:
        raise bad(f"{path}.field", f"unknown field {f!r}", f"use one of {list(FIELDS)}")
    if op not in FIELDS[f]:
        raise bad(f"{path}.op", f"op {op!r} does not apply to {f!r}", f"use one of {list(FIELDS[f])}")
    if "value" not in rules:
        raise bad(f"{path}.value", "missing value", "add a value")
    v = rules["value"]
    ok = {"between": isinstance(v, list) and len(v) == 2, "in": isinstance(v, list), "within_days": isinstance(v, (int, float)),
          "compatible_with": isinstance(v, dict) and ("bpm" if f == "bpm" else "camelot") in v}.get(op, True)
    if not ok:
        raise bad(f"{path}.value", f"value does not fit op {op!r}", {"between": "[low, high]", "in": "a list",
                  "within_days": "a number of days", "compatible_with": "bpm: {bpm, percent}; camelot: {camelot, max_distance}"}[op])


def _one(fact, op: str, v, field: str, now: float) -> bool:
    if fact is None:
        return False
    if field == "tag":
        return any(x in fact for x in v) if op == "in" else v in fact
    if op == "contains":
        return str(v).lower() in str(fact).lower()
    if op == "within_days":
        return now - fact <= v * DAY
    if op == "compatible_with":
        if field == "bpm":
            return min(abs(fact * m / v["bpm"] - 1) for m in (0.5, 1.0, 2.0)) * 100 <= v.get("percent", 6)
        return camelot_relation(v["camelot"], fact)[0] <= v.get("max_distance", 1)
    if isinstance(fact, str):
        fact, v = fact.lower(), ([x.lower() for x in v] if isinstance(v, list) else str(v).lower())
    return {"eq": lambda: fact == v, "ne": lambda: fact != v, "in": lambda: fact in v, "gte": lambda: fact >= v, "lte": lambda: fact <= v,
            "between": lambda: v[0] <= fact <= v[1]}[op]()


def match(rules: dict, facts: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if "all" in rules:
        return all(match(r, facts, now) for r in rules["all"])
    if "any" in rules:
        return any(match(r, facts, now) for r in rules["any"])
    return _one(facts.get(rules["field"]), rules["op"], rules["value"], rules["field"], now)
