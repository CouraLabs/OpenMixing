from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.units import SR

MAX_TAIL_S = 12.0
EDGE = int(0.005 * SR)


@dataclass
class FxSpec:
    name: str
    mode: str
    description: str
    uses_beats: str
    params: dict


CATALOG: dict[str, FxSpec] = {s.name: s for s in [
    FxSpec("echo", "send", "Damped feedback delay (Serato Echo)", "delay time", {"feedback": 0.5, "damping_hz": 6000}),
    FxSpec("echo_out", "send", "Echo with the dry signal cut: the deck 'echoes out' (Serato Echo Out)", "delay time",
           {"feedback": 0.65, "damping_hz": 5000}),
    FxSpec("delay", "send", "Clean feedback delay (Serato Delay)", "delay time", {"feedback": 0.35}),
    FxSpec("ping_pong", "send", "Stereo ping-pong delay (Serato Ping Pong Delay)", "delay time",
           {"feedback": 0.55, "damping_hz": 7000}),
    FxSpec("reverb", "send", "Room reverb (Serato Reverb)", "decay time", {"damping": 0.5}),
    FxSpec("epic_reverb", "send", "Huge, long reverb (Serato Epic Reverb)", "decay time", {"damping": 0.3}),
    FxSpec("flanger", "insert", "Jet flanger (Serato Flanger)", "LFO period", {"depth": 0.7, "feedback": 0.6}),
    FxSpec("phaser", "insert", "Phaser (Serato Phaser)", "LFO period", {"depth": 0.8, "feedback": 0.4}),
    FxSpec("chorus", "insert", "Chorus (Serato Chorus)", "LFO period", {"depth": 0.3}),
    FxSpec("lfo_filter", "insert", "Low-pass filter swept by a tempo-synced LFO (Serato LFO Filter)", "LFO period",
           {"min_hz": 250, "max_hz": 12000, "resonance": 0.4}),
    FxSpec("tremolo", "insert", "Tempo-synced volume LFO (Serato Tremolo)", "LFO period", {"depth": 1.0}),
    FxSpec("transform", "insert", "Rhythmic gate (Serato Transform)", "gate period", {"duty": 0.5}),
    FxSpec("repeater", "insert", "Beat repeater / stutter: captures the first slice and repeats it (Serato Repeater)",
           "slice length", {}),
    FxSpec("reverse", "insert", "Plays the region backwards in slices", "slice length", {}),
    FxSpec("brake", "insert", "Turntable stop (Serato Braker); silent once stopped", "stop time", {"curve": 1.0}),
    FxSpec("bitcrush", "insert", "Bit/sample-rate reduction (Serato Bitcrush)", "", {"bits": 6, "downsample": 4}),
    FxSpec("distortion", "insert", "Overdrive (Serato Distortion)", "", {"drive_db": 20}),
    FxSpec("compressor", "insert", "Soft-knee compressor with peak detection (ratio 1 = off)", "",
           {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0, "knee_db": 6.0, "makeup_db": 0.0}),
    FxSpec("sidechain_pump", "insert", "Kick-less sidechain pumping: the level dips on every sync period and recovers", "pump period",
           {"depth_db": 9.0, "release_beats": 0.8}),
]}


@dataclass
class FxOut:
    y: np.ndarray
    dry_cut: bool = False
