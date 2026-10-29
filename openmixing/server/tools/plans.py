from __future__ import annotations

import json

from ... import plan as planlib
from ..app import mcp, tool
from ..state import LOCK, STATE, cur, lib, save
from ..views import clip_row, jsonable


@tool()
def get_plan_schema() -> dict:
    """The versioned JSON schema of a set plan (units, ranges, curve types, effect names). Write a plan, `validate_plan` it, then
    `apply_plan`. Plans are data only: every operation is allowlisted and range-checked."""
    return {"version": planlib.PLAN_VERSION, "schema": planlib.plan_schema()}


@tool()
def validate_plan(plan: dict) -> dict:
    """Check a plan without touching the current set. Returns every problem at once: {code, field (JSON path), message, fix}."""
    errors = planlib.validate_plan(plan, lib())
    return {"version": planlib.PLAN_VERSION, "ok": not errors, "n_errors": len(errors), "errors": errors}


@tool()
def apply_plan(plan: dict, dry_run: bool = False) -> dict:
    """Replace the WHOLE current set with the plan (atomic: if the plan has any error, nothing changes and all errors are returned). A
    candidate plan from suggest_transitions is only a 2-clip fragment: applying it discards every other clip, so export_plan first and merge.
    dry_run=True only validates and reports the resulting timeline."""
    model, errors = planlib.build_plan(plan, lib())
    if errors:
        return {"version": planlib.PLAN_VERSION, "applied": False, "n_errors": len(errors), "errors": errors}
    with LOCK:
        if not dry_run:
            STATE["set"] = model
            save()
        saved = STATE["set"]
        STATE["set"] = model
        try:
            rows = [clip_row(c) for c in model.clips]
        finally:
            STATE["set"] = saved
    return jsonable({"version": planlib.PLAN_VERSION, "applied": not dry_run, "set": model.name, "clips": rows})


@tool()
def export_plan() -> dict:
    """The current set as a plan document (round-trips through apply_plan)."""
    return jsonable(planlib.export_plan(cur()))


@mcp.resource("openmixing://plan-schema")
def plan_schema_resource() -> str:
    """The set plan JSON schema."""
    return json.dumps(planlib.plan_schema(), indent=1)
