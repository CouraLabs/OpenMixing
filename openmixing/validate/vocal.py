from __future__ import annotations

import numpy as np

VOCAL_DB = -40.0


def vocal_collision(vocal_db: dict, method: dict, conf: dict, block_sec: float, t0: float) -> dict:
    ids = list(vocal_db)
    used = sorted({method[i] for i in ids})
    base = {"threshold_db": VOCAL_DB, "method": used[0] if len(used) == 1 else "mixed" if used else "none",
            "methods": {i: method[i] for i in ids}}
    if len(ids) < 2:
        return {"seconds_overlap": 0.0, "segments": [], "max_level_db": None, "confidence": 1.0 if ids else 0.0, **base}
    n = min(len(v) for v in vocal_db.values())
    db = np.array([np.asarray(vocal_db[i])[:n] for i in ids])
    on = db > VOCAL_DB
    both = on.sum(axis=0) >= 2
    second = np.sort(db, axis=0)[-2]
    segs, i = [], 0
    while i < n:
        if not both[i]:
            i += 1
            continue
        j = i
        while j < n and both[j]:
            j += 1
        segs.append({"start_sec": round(t0 + i * block_sec, 2), "end_sec": round(t0 + j * block_sec, 2),
                     "clips": [c for k, c in enumerate(ids) if on[k, i:j].any()], "level_db": round(float(second[i:j].max()), 1)})
        i = j
    involved = [conf[c] for s in segs for c in s["clips"]]
    return {"seconds_overlap": round(float(both.sum() * block_sec), 2), "segments": segs[:8],
            "max_level_db": round(float(second[both].max()), 1) if both.any() else None,
            "confidence": round(float(min(involved)), 3) if involved else 1.0, **base}
