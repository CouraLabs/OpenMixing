from __future__ import annotations

from typing import Literal

from ...core.errors import OpenMixingError
from ...core.units import BAR
from ...dsp import sample
from ...model.events import Trigger, new_id
from ...model.sampler import KINDS, SampleSlot, check_voices
from ...model.tempo import TempoMap
from ..app import Len, Pos, tool
from ..state import LOCK, at_of, clip_of, cur, info, len_of, lib, save
from .triggers import checked_gain, checked_params

SampleKind = Literal[*KINDS]


def slot_row(name: str, s: SampleSlot) -> dict:
    uses = sum(t.slot == name for c in cur().clips for t in c.triggers)
    return {"slot": name, **s.to_dict(), "seconds": round(lib().sample_seconds(s.path), 3), "triggers": uses}


def natural_beats(c, s: SampleSlot, start: float, sec: float) -> float:
    if s.kind == "loop" and s.root_bpm:
        return sec * s.root_bpm / 60.0
    tm = TempoMap(c, float(info(c.track)["bpm"]))
    return float(tm.sec_to_beat(tm.beat_to_sec(start) + sec) - start)


@tool()
def load_sample(slot: str, path: str, kind: SampleKind = "oneshot", root_bpm: float | None = None, gain_db: float = 0.0, start_sec: float = 0.0,
                end_sec: float | None = None, pitch_semitones: float = 0.0, choke_group: str | None = None) -> dict:
    """Load a one-shot or loop into a named sampler slot of the current set. `path` is a file inside the tracks folder or .openmixing/samples/
    (anything else is refused with path_not_allowed). It is decoded, trimmed to start_sec..end_sec, peak-normalized and stored as a
    content-addressed 48 kHz stereo wav under .openmixing/samples/. One-shots may last 30 s, loops 120 s. A loop without root_bpm gets its
    tempo detected (snapped to a whole number of beats) and then follows the clip tempo when triggered. Slots with the same choke_group
    cut each other when a trigger has choke=1. Reloading a slot replaces it."""
    if not slot.strip():
        raise OpenMixingError("wrong_type", "the slot name is empty", "slot", "use a name such as kick or vox_hit")
    checked_gain(gain_db)
    if not -24.0 <= pitch_semitones <= 24.0:
        raise OpenMixingError("out_of_range", f"pitch_semitones {pitch_semitones} is outside [-24, 24]", "pitch_semitones", "use a smaller shift")
    with LOCK:
        got = lib().import_sample(path, kind, root_bpm, start_sec, end_sec)
        cur().sampler[slot] = s = SampleSlot(got["path"], kind, got["root_bpm"], float(gain_db), float(pitch_semitones), choke_group)
        save()
    return slot_row(slot, s)


@tool()
def list_samples() -> dict:
    """The sampler slots of the current set: file, kind, root_bpm, gain, pitch, choke group, length in seconds and how many triggers use each."""
    with LOCK:
        return {"slots": [slot_row(k, s) for k, s in cur().sampler.items()], "fits": list(sample.FITS)}


@tool()
def remove_sample(slot: str) -> dict:
    """Remove a sampler slot. Refused while triggers still use it (remove them with remove_trigger first)."""
    with LOCK:
        if slot not in cur().sampler:
            raise OpenMixingError("unknown_slot", f"no sampler slot {slot!r}", "slot", f"slots: {sorted(cur().sampler)}")
        used = [(c.id, t.id) for c in cur().clips for t in c.triggers if t.slot == slot]
        if used:
            raise OpenMixingError("slot_in_use", f"{len(used)} trigger(s) still play {slot!r}: {used[:6]}", "slot", "remove_trigger them first")
        del cur().sampler[slot]
        save()
    return {"removed": slot, "slots": sorted(cur().sampler)}


@tool()
def trigger_sample(clip: str, slot: str, at_bar: Pos, bars: Len | None = None, gain_db: float = -6.0, pitch_semitones: float = 0.0,
                   fit: Literal[*sample.FITS] = "none", velocity: float = 0.0, choke: bool = False) -> dict:
    """Play a sampler slot on a deck (post-fader, before the deck fx) starting at `at_bar`. It lasts `bars` (default: the sample's own length at
    the clip tempo; a loop's length in beats at its root_bpm). fit="none" plays it at its own speed (loops follow the clip tempo) and cuts it at
    the end, "stretch_to_beats" time-stretches it to exactly `bars`, "loop_for_duration" repeats it to fill `bars`. pitch_semitones adds to the
    slot's shift, velocity is dB on top of gain_db (relative to the deck level, see list_generators), choke=True cuts earlier voices of the same
    choke_group. At most 16 voices may sound at once on one deck."""
    gain = checked_gain(gain_db)
    p = checked_params("sample", {"pitch_semitones": pitch_semitones, "fit": float(sample.FITS.index(fit)), "velocity": velocity,
                                  "choke": float(choke)})
    with LOCK:
        if slot not in cur().sampler:
            raise OpenMixingError("unknown_slot", f"no sampler slot {slot!r}", "slot", f"slots: {sorted(cur().sampler)}")
        c, s = clip_of(clip), cur().sampler[slot]
        start = at_of(c, at_bar) * BAR
        if bars is None:
            sec = lib().sample_seconds(s.path)
            dur = max(natural_beats(c, s, start, sec), 1 / 16)
        else:
            dur = len_of(bars) * BAR
        if start < -1e-9 or start > c.length_beats:
            raise OpenMixingError("out_of_range", f"the sample would start at beat {start:g}, outside the clip", "at_bar", "move it inside the clip")
        t = Trigger(new_id("tg"), "sample", start, dur, "start", gain, p, slot)
        check_voices([(x.start, x.start + x.duration_beats) for x in [*c.triggers, t] if x.kind == "sample"])
        c.triggers.append(t)
        save()
    return {"clip": clip, "trigger_id": t.id, "starts_at_bar": t.start / BAR, "ends_at_bar": (t.start + t.duration_beats) / BAR}
