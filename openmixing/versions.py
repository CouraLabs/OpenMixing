"""Schema and analysis versions reported in tool responses."""
from __future__ import annotations

from importlib import metadata


def versions() -> dict:
    from . import candidates, plan, validate
    from .library import ANALYSIS_VERSION
    try:
        pkg = metadata.version("openmixing")
    except metadata.PackageNotFoundError:
        pkg = None
    return {"openmixing": pkg, "analysis": ANALYSIS_VERSION, "plan_schema": plan.PLAN_VERSION, "validation": validate.VERSION,
            "candidates": candidates.VERSION}
