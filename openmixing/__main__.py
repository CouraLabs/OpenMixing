"""`openmixing` = MCP stdio server for the tracks folder in the current directory.
`openmixing ui [--dir D] [--port P]` serves the player for already-rendered sets without an agent."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="openmixing", description=__doc__)
    ap.add_argument("command", nargs="?", choices=["mcp", "ui"], default="mcp")
    ap.add_argument("--dir", help="tracks folder (default: current directory or $OPENMIXING_DIR)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.dir:
        os.environ["OPENMIXING_DIR"] = args.dir
    if args.command == "ui":
        import threading
        import webbrowser

        from .library import Library
        from .ui.server import start

        info = start(Library(), args.port)
        print(f"OpenMixing player: {info['url']}", file=sys.stderr)
        webbrowser.open(info["url"])
        threading.Event().wait()
    else:
        from .server import main as run

        run()


if __name__ == "__main__":
    main()
