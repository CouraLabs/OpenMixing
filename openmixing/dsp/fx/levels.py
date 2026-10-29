from __future__ import annotations

from .. import dynamics
from .catalog import FxOut


def compressor(region, beat_sec, beats, p, name):
    return FxOut(dynamics.compress(region, p["threshold_db"], max(p["ratio"], 1.0), p["attack_ms"], p["release_ms"], p["knee_db"], p["makeup_db"]))


def sidechain_pump(region, beat_sec, beats, p, name):
    return FxOut(dynamics.pump(region, beat_sec, p["depth_db"], p["release_beats"], beats))
