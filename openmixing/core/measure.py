from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .errors import OpenMixingError


def measure(path: Path) -> dict:
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    i = r.stderr.rfind("Summary:")
    if r.returncode or i < 0:
        raise OpenMixingError("io_error", f"loudness measurement failed for {Path(path).name}: {r.stderr.strip()[-300:]}", "path",
                              "check the file is valid audio")
    t = r.stderr[i:]
    num = lambda pat: (float(m.group(1)) if (m := re.search(pat, t)) else None)
    return {"integrated_lufs": num(r"I:\s+(-?[\d.]+) LUFS"), "loudness_range_lu": num(r"LRA:\s+(-?[\d.]+) LU"),
            "true_peak_dbtp": num(r"Peak:\s+(-?[\d.]+) dBFS")}
