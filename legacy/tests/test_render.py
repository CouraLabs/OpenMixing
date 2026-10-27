import numpy as np
import pytest

from openmixing.models import SR, TransitionPlan
from openmixing.render.mix import render_mix, segments
from openmixing.render.transition import PlanError, render_transition
from openmixing.synth import make_track


def plan(**kw):
    d = dict(from_track="a", to_track="b", from_bar=8, to_bar=0, length_bars=8, target_bpm=120.0,
             tempo_ramp={"hold_bars": 4, "ramp_bars": 4},
             eq_automation=[{"bar": 0, "deck": "to", "bass": "kill"}, {"bar": 4, "deck": "to", "bass": "full"},
                            {"bar": 4, "deck": "from", "bass": "kill"}],
             crossfade={"curve": "equal_power", "bars": [0, 8]})
    d.update(kw)
    return TransitionPlan.from_dict(d)


def onsets(x, thresh=0.2, min_gap=0.2):
    env = np.abs(x)
    idx = np.flatnonzero(env > thresh)
    out, last = [], -10**9
    for i in idx:
        if i - last > min_gap * SR:
            out.append(i)
        last = i
    return np.array(out)


def test_length_and_window_exact():
    a, b = make_track("a", 120, 24), make_track("b", 126, 24)
    r = render_transition(a, b, plan())
    assert r.window.shape == (2, 8 * 2 * SR)  # 8 bars @120 BPM = 16 s
    assert r.tail.shape[0] == 2


def test_incoming_deck_beat_aligned_with_outgoing_and_continuous_after_window():
    a, b = make_track("a", 120, 24, click_hz=1000), make_track("b", 126, 24, click_hz=3000)
    # Incoming deck only (fader fully to B from sample 0), no EQ so the click stays intact.
    p = plan(eq_automation=[], crossfade={"curve": "linear", "bars": [0, 0]})
    r = render_transition(a, b, p)
    w = r.window[0]
    beat = 0.5 * SR
    on = onsets(w)
    assert len(on) >= 30
    # hold phase (first 4 bars = 16 beats): onsets sit on A's 120 BPM grid
    hold = on[:16]
    err = hold - np.arange(16) * beat
    assert np.abs(err).max() < 0.002 * SR, err
    # splice back to native B: onsets continue with no phase glitch across the window end
    full = np.concatenate([r.window, b.slice(r.b_resume, b.n_samples)], axis=1)[0]
    on2 = onsets(full)
    k = np.searchsorted(on2, r.window.shape[1])
    gap = on2[k] - on2[k - 1]
    assert abs(gap - 60 / 126 * SR) < 0.01 * SR, gap
    ibi = np.diff(on2[k:k + 8])
    assert np.abs(ibi - 60 / 126 * SR).max() < 0.004 * SR


def test_outgoing_window_sample_aligned_with_source():
    a, b = make_track("a", 120, 24), make_track("b", 120, 24)
    r = render_transition(a, b, plan(tempo_ramp={"hold_bars": 8, "ramp_bars": 0}, eq_automation=[],
                                     crossfade={"curve": "linear", "bars": [8, 8]}))
    src = a.slice(r.a_start, r.a_start + r.window.shape[1])
    # Window equals the source exactly for the outgoing deck (fader fully on A until the last bar)
    n = 4 * SR
    ref, got = src[0, :n], r.window[0, :n]
    corr = np.correlate(got[: 2 * SR], ref[: 2 * SR], "full")
    assert corr.argmax() - (2 * SR - 1) == 0


def test_plan_errors_are_structured():
    a, b = make_track("a", 120, 24), make_track("b", 140, 24)
    with pytest.raises(PlanError) as e:
        render_transition(a, b, plan())
    assert any(x["field"] == "target_bpm" for x in e.value.errors)


def test_echo_tail_and_preloop():
    a, b = make_track("a", 120, 24), make_track("b", 120, 24)
    p = plan(tempo_ramp={"hold_bars": 8, "ramp_bars": 0}, pre_loop={"bars": 2, "repeats": 2},
             fx=[{"type": "echo_out", "deck": "from", "bars": [6, 8]},
                 {"type": "lowpass_sweep", "deck": "from", "bars": [0, 6], "start_hz": 18000, "end_hz": 800}])
    r = render_transition(a, b, p)
    assert r.pre_loop.shape[1] == 2 * 2 * 2 * SR
    assert np.abs(r.tail).max() > 0.01
    mix = render_mix(a, b, p)
    assert np.isfinite(mix).all() and np.abs(mix).max() <= 1.0
    assert sum(s.shape[1] for s in segments(a, b, r)) > r.window.shape[1]
