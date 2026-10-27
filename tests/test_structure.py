import numpy as np

from openmixing.analysis.features import structure


def _bf(blocks, n, seed=0):
    """Per-bar features that change at `blocks` starts; fill bars one before each change already look like the next block."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4 + 64 + 12))
    level = np.zeros(n)
    for i, (a, v) in enumerate(blocks):
        level[a:] = v
    X[:, 0] = -30 + 10 * level
    X[:, 1] = -35 + 12 * (level % 2)
    X[:, 4:68] = level[:, None] * rng.standard_normal(64)[None] + 0.01 * rng.standard_normal((n, 64))
    return {"n_bars": n, "X": X, "bar_sec": 1.9}


def test_boundaries_are_pulled_onto_the_four_bar_grid():
    # real changes at 16 / 32 / 48, but the fill bar before each (15 / 31 / 47) already looks like the next section
    bf = _bf([(0, 1.0), (15, 2.0), (31, 3.0), (47, 4.0)], 56)
    st = structure(bf, 0.5)
    starts = [s["start_bar"] for s in st["sections"]]
    assert starts[0] == 0 and all(a % 4 == 0 for a in starts), starts
    assert {s["raw_start_bar"] for s in st["sections"]} >= {15, 31, 47}


def test_aligned_boundaries_are_untouched():
    st = structure(_bf([(0, 1.0), (16, 2.0), (32, 3.0)], 48), 0.5)
    assert all(s["raw_start_bar"] == s["start_bar"] for s in st["sections"])
