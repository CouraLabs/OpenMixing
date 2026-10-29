from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..core.errors import structured
from ..model.curves import CURVES
from .guide import GUIDE

mcp = FastMCP("openmixing", instructions=GUIDE)


def tool():
    def deco(fn):
        mcp.tool()(structured(fn))
        return fn
    return deco


Beats = Annotated[str | float, Field(description="Tempo-sync multiplier in beats: one of 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32")]
Pos = Annotated[float | str, Field(description="Position in the clip's playback: output bars (0 = clip start), \"32 beats\", "
                                              "or a place in the source track (\"cue:drop\", \"section:breakdown+4\", \"95.5s\"), "
                                              "converted with the clip's src_start_bar")]


Curve = Literal[*CURVES]
Len = Annotated[float | str, Field(description="Length in bars, or \"N beats\"")]


@mcp.resource("openmixing://guide")
def guide_resource() -> str:
    """The DJ playbook."""
    return GUIDE


@mcp.prompt()
def dj_set(goal: str = "a smooth 45 minute house set") -> str:
    """Start a DJ session."""
    return f"{GUIDE}\n\nTask: build {goal} from the tracks in this folder. Analyze, order, mix with apply_transition, audit_set, render, open_player."


def main() -> None:
    mcp.run(transport="stdio")
