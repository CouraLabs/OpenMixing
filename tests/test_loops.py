import numpy as np
import pytest

from openmixing.core.units import SR
from openmixing.engine import render_clip, segments
from openmixing.model import Clip, LoopEvent, TempoMap

BPM, BARS = 120.0, 8
BAR_SEC = 4 * 60 / BPM
INFO = {"bpm": BPM, "first_downbeat": 0.0, "gain_db": 0.0}


def freq(bar: int) -> float:
    return 220.0 + 110.0 * bar


def tone_per_bar() -> np.ndarray:
    n = int(BAR_SEC * SR)
    t = np.arange(n) / SR
    y = np.concatenate([np.sin(2 * np.pi * freq(k) * t) for k in range(BARS)]).astype(np.float32) * 0.3
    return np.stack([y, y])


def source_bars(clip: Clip) -> list[int]:
    tm = TempoMap(clip, BPM)
    out = render_clip(clip, INFO, tone_per_bar(), tm)[0]
    n = int(BAR_SEC * SR)
    bars = []
    for k in range(int(clip.length_beats // 4)):
        seg = out[k * n + n // 4: (k + 1) * n - n // 4]
        f = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
        bars.append(int(round((f - 220.0) / 110.0)))
    return bars


def looped(roll: bool) -> Clip:
    return Clip("c", "t", 0.0, 24.0, loops=[LoopEvent("l", 8.0, 4.0, 12.0, roll)])


def test_loop_repeats_its_bar_then_continues_after_it():
    assert source_bars(looped(False)) == [0, 1, 2, 2, 2, 3]


def test_loop_roll_resumes_where_the_track_would_be():
    assert source_bars(looped(True)) == [0, 1, 2, 2, 2, 5]


@pytest.mark.parametrize("roll, after", [(False, 12.0), (True, 20.0)])
def test_segments_source_after_loop(roll, after):
    segs = segments(looped(roll))
    assert segs[1] == ("loop", 8.0, 20.0, 8.0, 4.0)
    assert segs[2][3] == after


def test_loop_length_not_a_multiple_of_size_resumes_after_the_partial_pass():
    clip = Clip("c", "t", 0.0, 24.0, loops=[LoopEvent("l", 8.0, 4.0, 6.0, False)])
    assert segments(clip) == [("play", 0.0, 8.0, 0.0), ("loop", 8.0, 14.0, 8.0, 4.0), ("play", 14.0, 24.0, 10.0)]
    roll = Clip("c", "t", 0.0, 24.0, loops=[LoopEvent("l", 8.0, 4.0, 6.0, True)])
    assert segments(roll)[-1] == ("play", 14.0, 24.0, 14.0)


def test_loop_is_truncated_at_the_clip_end():
    clip = Clip("c", "t", 0.0, 12.0, loops=[LoopEvent("l", 8.0, 4.0, 12.0, False)])
    assert segments(clip) == [("play", 0.0, 8.0, 0.0), ("loop", 8.0, 12.0, 8.0, 4.0)]
