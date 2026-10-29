import resource
import sys
import time

import pytest
import soundfile as sf
from synth import make_track

from openmixing import engine
from openmixing import server as S
from openmixing.library import Library

TIME_BUDGET_S = 300
RSS_BUDGET_MB = 3000


def rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20 if sys.platform == "darwin" else 1 << 10)


@pytest.fixture
def folder(tmp_path):
    make_track(tmp_path / "a.wav", bpm=120, bars=32)
    S.STATE.update(lib=Library(tmp_path), set=None, jobs=None)
    S.override_track("a.wav", bpm=120.0, first_downbeat_sec=0.0)
    S.new_set("perf")
    yield tmp_path
    S.STATE.update(lib=None, set=None, jobs=None)


def test_ten_minute_set_with_all_event_types_stays_within_budget(folder):
    S.load_sample("hit", "a.wav", start_sec=0.0, end_sec=0.5)
    prev = None
    for i in range(10):
        c = S.add_clip("a.wav", bars=32)["clip"]
        S.add_rate(c, 10, 1, "spinback", slip=True)
        S.add_rate(c, 14, 0.5, "brake")
        S.add_roll(c, 28, 2)
        S.add_trigger(c, "riser", 24, 4)
        S.add_trigger(c, "impact", 31, 1)
        S.trigger_sample(c, "hit", 8, 0.25)
        S.add_fx(c, "echo", 16, 2, beats=1.0, chain=[{"fx": "compressor"}, {"fx": "echo", "beats": 0.5}])
        S.add_loop(c, 20, 4, repeats=2)
        S.set_eq(c, "low", 4, -6, ramp_bars=2)
        if prev:
            S.add_duck(c, prev, 0, 4)
        prev = c
    out = folder / ".openmixing" / "renders" / "perf.wav"
    t0 = time.time()
    meta = engine.render_set(S.cur(), S.info, out, fmt="wav")
    took = time.time() - t0
    assert meta["duration_sec"] > 600
    assert sf.info(out).duration == pytest.approx(meta["duration_sec"], abs=0.5)
    assert took < TIME_BUDGET_S, f"rendered in {took:.0f}s"
    assert rss_mb() < RSS_BUDGET_MB
