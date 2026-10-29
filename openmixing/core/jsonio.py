from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .errors import OpenMixingError


def read_json(path: Path, default=None, strict: bool = False):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        if strict:
            raise OpenMixingError("io_error", f"{Path(path).name} is unreadable or corrupt: {e}", fix=f"repair or delete {path} and retry") from e
        return default


def write_json(path: Path, data, indent: int | None = None) -> None:
    p = Path(path)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(json.dumps(data, indent=indent))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
