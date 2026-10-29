from __future__ import annotations

import numpy as np

from .confidence import attach, unit
from .key import rank_keys


def harmony_block(bf: dict, st: dict) -> dict:
    X, phrase = bf["X"], st.get("phrase_bars")
    if not st["sections"] or not phrase:
        return {"chroma_bar": [], "key_by_phrase": [], "confidence": 0.0, "confidence_components": {"phrases": 0.0}}
    chroma = X[:, 68:80]
    keys = []
    for a in range(0, bf["n_bars"] - phrase + 1, phrase):
        r = rank_keys(chroma[a:a + phrase].mean(0))
        keys.append(attach({"start_bar": a, "bars": phrase, "key": r[0]["key"], "camelot": r[0]["camelot"]},
                           {"margin": unit(r[0]["score"] - r[1]["score"], 0, 0.125), "correlation": unit(r[0]["score"], 0.4, 0.9)}))
    return attach({"chroma_bar": np.round(chroma, 3).tolist(), "key_by_phrase": keys},
                  {"phrases": float(np.mean([k["confidence"] for k in keys])) if keys else 0.0})
