"""CLI for the engine slice: `python -m openmixing.cli demo|render|play`."""
from __future__ import annotations

import argparse
import json

from .decode import decode
from .models import Track, TransitionPlan
from .planner.validator import validate_plan
from .render.mix import render_to_wav
from .synth import make_track


def _tracks(args) -> tuple[Track, Track]:
    if args.demo:
        return (make_track("a", 120, 40, bass_hz=55, pad_hz=220), make_track("b", 124, 40, click_hz=2000, bass_hz=62, pad_hz=330))
    a = Track("a", decode(args.a), args.bpm_a, args.offset_a)
    b = Track("b", decode(args.b), args.bpm_b, args.offset_b)
    return a, b


def _plan(args, a: Track) -> TransitionPlan:
    if args.plan:
        with open(args.plan) as f:
            return TransitionPlan.from_dict(json.load(f))
    return TransitionPlan.from_dict({
        "from_track": "a", "to_track": "b", "from_bar": a.n_bars - 40 if args.demo else a.n_bars - 36,
        "to_bar": 0, "length_bars": 32, "target_bpm": a.bpm,
        "tempo_ramp": {"hold_bars": 24, "ramp_bars": 8},
        "eq_automation": [{"bar": 0, "deck": "to", "bass": "kill"}, {"bar": 16, "deck": "to", "bass": "full"},
                          {"bar": 16, "deck": "from", "bass": "kill"}],
        "fx": [{"type": "lowpass_sweep", "deck": "from", "start_hz": 18000, "end_hz": 800, "bars": [8, 24]}],
        "crossfade": {"curve": "equal_power", "bars": [0, 32]},
    })


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="openmixing")
    ap.add_argument("command", choices=["render", "play"])
    ap.add_argument("a", nargs="?"), ap.add_argument("b", nargs="?")
    ap.add_argument("--demo", action="store_true", help="use synthetic click tracks")
    ap.add_argument("--bpm-a", type=float), ap.add_argument("--bpm-b", type=float)
    ap.add_argument("--offset-a", type=float, default=0.0, help="first downbeat (s)")
    ap.add_argument("--offset-b", type=float, default=0.0)
    ap.add_argument("--plan", help="TransitionPlan JSON file (default: a 32-bar EQ swap)")
    ap.add_argument("--out", default="mix.wav")
    args = ap.parse_args(argv)
    if not args.demo and not (args.a and args.b and args.bpm_a and args.bpm_b):
        ap.error("give A B --bpm-a --bpm-b (hand-entered grid for now), or --demo")

    a, b = _tracks(args)
    plan = _plan(args, a)
    errors = validate_plan(plan, a, b)
    if errors:
        raise SystemExit(json.dumps(errors, indent=2))
    if args.command == "render":
        render_to_wav(a, b, plan, args.out)
        print(f"wrote {args.out}")
    else:
        from .engine.player import play_set_in_process

        print("underruns:", play_set_in_process(a, b, plan))


if __name__ == "__main__":
    main()
