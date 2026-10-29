from __future__ import annotations

from ..model.clip import Clip
from ..model.setmodel import SetModel
from ..model.tempo import resolve_starts
from .clips import parse_clip, parse_start
from .context import Ctx
from .ducks import parse_ducks
from .intent import parse_intent
from .keymatch import resolve_key_matches
from .sampler import parse_sampler
from .schema import PLAN_VERSION, PLAN_VERSIONS, PlanError


def build_plan(plan, lib) -> tuple[SetModel | None, list[dict]]:
    cx = Ctx(lib)
    if not cx.obj(plan, "plan", ("version", "name", "notes", "clips", "intent", "ducks", "sampler"), ("version", "clips")):
        return None, cx.errors
    if plan["version"] not in PLAN_VERSIONS:
        cx.add("unsupported_version", "plan.version", f"plan version {plan['version']!r} is not supported",
               f"use {PLAN_VERSION!r} (or {PLAN_VERSIONS[0]!r})")
    if not isinstance(plan["clips"], list) or not plan["clips"]:
        cx.add("wrong_type", "plan.clips", "expected a non-empty list of clips")
        return None, cx.errors
    cx.slots = parse_sampler(cx, plan.get("sampler", {}))
    ids_all = {c["id"] for c in plan["clips"] if isinstance(c, dict) and isinstance(c.get("id"), str)}
    ids_all |= {f"c{n}" for n, c in enumerate(plan["clips"], 1) if isinstance(c, dict) and "id" not in c}
    ids: set[str] = set()
    clips: list[Clip] = []
    prev = None
    for i, c in enumerate(plan["clips"]):
        clip = parse_clip(cx, lib, c, i, ids, i + 1)
        if clip is None:
            prev = None
            continue
        parse_start(cx, c, i, clip, prev, ids_all)
        clips.append(clip)
        prev = clip
    resolve_key_matches(cx, lib, plan["clips"], clips)
    name = plan.get("name", "plan")
    if not isinstance(name, str) or not name:
        cx.add("wrong_type", "plan.name", "expected a non-empty string")
        name = "plan"
    model = SetModel(name, clips, str(plan.get("notes", "")), intent=parse_intent(cx, plan.get("intent")),
                     ducks=parse_ducks(cx, plan.get("ducks", []), clips), sampler=cx.slots)
    if not cx.errors:
        try:
            resolve_starts(model, lambda t: float(lib.effective(t)["bpm"]))
        except ValueError as e:
            cx.add("anchor_cycle", "plan.clips", str(e), "anchors must form a chain that starts at an at_sec clip")
    return (None, cx.errors) if cx.errors else (model, [])


def validate_plan(plan, lib) -> list[dict]:
    return build_plan(plan, lib)[1]


def apply_plan(plan, lib) -> SetModel:
    model, errors = build_plan(plan, lib)
    if errors:
        raise PlanError(errors)
    return model
