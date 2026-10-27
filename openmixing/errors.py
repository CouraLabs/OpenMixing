"""Structured tool errors: machine-readable code, the offending field and a suggested fix."""
from __future__ import annotations

import functools
import inspect
import json

from mcp.server.fastmcp.exceptions import ToolError


class OpenMixingError(Exception):
    def __init__(self, code: str, message: str, field: str | None = None, fix: str | None = None):
        super().__init__(message)
        self.code, self.message, self.field, self.fix = code, message, field, fix

    def payload(self) -> dict:
        return {"error": {"code": self.code, "field": self.field, "message": self.message, "fix": self.fix}}


FIXES = {
    "not_found": "list the available ids with list_tracks / get_set / list_sets and retry with one of them",
    "invalid_value": "check the parameter ranges in the tool description and retry",
    "missing_dependency": "install the optional extra named in the message (see README)",
    "io_error": "check the path is inside the tracks folder and readable",
}


def classify(e: Exception) -> OpenMixingError:
    if isinstance(e, OpenMixingError):
        return e
    msg = str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)
    if isinstance(e, (KeyError, FileNotFoundError)):
        code = "not_found"
    elif isinstance(e, (ValueError, TypeError)):
        code = "invalid_value"
    elif isinstance(e, ImportError):
        code = "missing_dependency"
    elif isinstance(e, OSError):
        code = "io_error"
    else:
        code = "internal_error"
    return OpenMixingError(code, msg, fix=FIXES.get(code))


def _fail(e: Exception):
    if isinstance(e, ToolError):
        raise e
    raise ToolError(json.dumps(classify(e).payload())) from e


def structured(fn):
    """Turn any exception raised by a tool into a JSON error payload."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            try:
                return await fn(*a, **kw)
            except Exception as e:
                _fail(e)
    else:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                _fail(e)
    return wrapper
