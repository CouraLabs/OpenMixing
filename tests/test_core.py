import numpy as np
import pytest

from openmixing.dsp import eq, fx, limiter
from openmixing.models import BEAT_DIVISIONS, SR, parse_beats


def test_parse_beats_accepts_all_divisions():
    for s in ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "32"]:
        assert parse_beats(s) in BEAT_DIVISIONS
    assert parse_beats(0.25) == 0.25
    with pytest.raises(ValueError):
        parse_beats("3")


def test_split_bands_flat_magnitude():
    x = np.random.default_rng(0).standard_normal((2, SR * 2)).astype(np.float32) * 0.1
    lo, mid, hi = eq.split_bands(x)
    X, Y = np.abs(np.fft.rfft(x[0])), np.abs(np.fft.rfft((lo + mid + hi)[0]))
    ratio = (Y / X)[200:20000]
    assert 0.9 < np.median(ratio) < 1.1 and ratio.min() > 0.2


def test_eq_kill_removes_band():
    t = np.arange(SR) / SR
    x = np.tile(np.sin(2 * np.pi * 60 * t).astype(np.float32), (2, 1))
    kill = np.full(SR, eq.KILL_DB)
    y = eq.apply_eq(x, kill, None, None)
    assert np.abs(y[:, SR // 2:]).max() < 0.05 * np.abs(x).max()


def test_limiter_respects_ceiling():
    x = (np.random.default_rng(1).standard_normal((2, SR * 2)) * 0.8).astype(np.float32)
    y = limiter.limit(x, -1.5)
    assert np.abs(y).max() <= 10 ** (-1.5 / 20) + 1e-3


def test_fx_unknown_name_rejected():
    assert "flanger" in fx.CATALOG and "bad" not in fx.CATALOG


@pytest.mark.parametrize("name", list(fx.CATALOG))
@pytest.mark.parametrize("beats", [1 / 16, 1.0, 32.0])
def test_every_fx_renders_finite(name, beats):
    region = (np.random.default_rng(2).standard_normal((2, SR // 2)) * 0.1).astype(np.float32)
    out = fx.render(name, region, 60 / 124, beats, {})
    assert out.y.shape[0] == 2 and np.isfinite(out.y).all()
