import itertools
import random

import numpy as np
import pytest
from synth import make_track

from openmixing import plan as P
from openmixing import setbuilder as SB
from openmixing.core.errors import OpenMixingError
from openmixing.library import Library
from openmixing.mixing import camelot_relation
from openmixing.setbuilder import arcs as A
from openmixing.setbuilder.constraints import Constraints
from openmixing.setbuilder.diagnose import infeasible
from openmixing.setbuilder.search import beam, improve
from openmixing.setbuilder.trim import fit

KEYS = ["8A", "9A", "7A", "8B", "9B", "10A"]


def items(n, seed=1, dur=300.0):
    r = random.Random(seed)
    out = []
    for i in range(n):
        k = r.choice(KEYS)
        out.append({"id": f"t{i:02d}", "file": f"t{i:02d}.wav", "bpm": 120 + r.choice([0, 2, 4, 6]), "camelot": k, "num": int(k[:-1]),
                    "energy": float(1 + i * 9 / max(n - 1, 1)) if seed == 1 else float(r.randint(2, 9)), "dur": dur, "artist": f"a{i % 4}"})
    return out


def scorer(its, cons=None, arc="build", target=None, **kw):
    ar = A.make(its, arc, kw.pop("tempo", None), None, False, False, 8)
    return SB.Scorer(its, cons or Constraints(), ar, None, target or sum(i["dur"] for i in its) * 0.8)


def test_presets_and_errors():
    for name in SB.PRESETS:
        pts = A.points(name, 2, 8, "energy_arc")
        assert min(p[1] for p in pts) >= 2 and max(p[1] for p in pts) <= 8
    with pytest.raises(OpenMixingError):
        A.points("nope", 1, 10, "energy_arc")
    with pytest.raises(OpenMixingError):
        A.points([[0, 1]], 1, 10, "energy_arc")


def test_tempo_and_key_arcs_are_independent_by_default():
    its = items(6)
    off = A.make(its, "build", None, None, False, False, 8)
    assert off["tempo"] is None and off["key"] is None
    on = A.make(its, "build", None, None, True, False, 8)
    assert on["tempo"] is not None and on["key"] is None
    both = A.make(its, "build", [[0, 120], [1, 126]], None, True, True, 8)
    assert both["tempo"][-1][1] == 126 and both["key"] is not None


def test_planted_arc_and_key_cap():
    its = items(12)
    sc = scorer(its, Constraints(max_key_distance=2), "build", target=2400.0)
    done, _ = beam(sc)
    seq = improve(sc, done[0][1])
    assert sc.violations(seq) == []
    es = [its[j]["energy"] for j in seq]
    assert np.corrcoef(es, np.linspace(0, 1, len(es)))[0, 1] >= 0.9
    for a, b in zip(seq, seq[1:]):
        assert camelot_relation(its[a]["camelot"], its[b]["camelot"])[0] <= 2


def test_beam_within_5_percent_of_brute_force():
    its = items(7, seed=3)
    sc = scorer(its, arc="wave", target=sum(i["dur"] for i in its))
    done, _ = beam(sc)
    best = min(sc.total(p) for p in itertools.permutations(range(7)))
    assert done[0][0] <= best * 1.05 + 1e-6


def test_hard_constraints_never_violated():
    its = items(10, seed=5)
    for s in range(30):
        r = random.Random(s)
        ids = [i["id"] for i in its]
        cons = Constraints(start_with=r.choice(ids) if s % 3 == 0 else None, must_include=r.sample(ids, 2),
                           must_not_follow=[r.sample(ids, 2) for _ in range(4)], max_tracks=r.choice([None, 6, 8]),
                           max_bpm_jump_percent=r.choice([None, 3.0]), min_gap_between_artist_repeats=r.choice([0, 1]),
                           max_key_distance=r.choice([None, 2]))
        sc = scorer(its, cons, "plateau", target=1500.0)
        done, _ = beam(sc)
        for _, seq in done[:5]:
            assert sc.violations(seq) == []
            assert len(set(seq)) == len(seq)
            assert set(sc.idx[t] for t in cons.must_include) <= set(seq)
            if cons.start_with:
                assert its[seq[0]]["id"] == cons.start_with
            if cons.max_tracks:
                assert len(seq) <= cons.max_tracks
            for a, b in zip(seq, seq[1:]):
                assert [its[a]["id"], its[b]["id"]] not in cons.must_not_follow
                if cons.min_gap_between_artist_repeats:
                    assert its[a]["artist"] != its[b]["artist"]
                if cons.max_bpm_jump_percent:
                    assert abs(sc.pair(a, b)["tempo"]["stretch_percent"]) <= cons.max_bpm_jump_percent


def test_end_with_and_fixed_positions():
    its = items(8, seed=2)
    cons = Constraints(end_with="t03", fixed_positions={1: "t05"})
    sc = scorer(its, cons, "plateau", target=1200.0)
    done, _ = beam(sc)
    assert done and all(its[s[-1]]["id"] == "t03" and its[s[1]]["id"] == "t05" for _, s in done)


def test_infeasible_names_the_constraint():
    its = items(5)
    for i, it in enumerate(its):
        it["camelot"], it["num"] = ["1A", "4A", "7A", "10A", "6B"][i], [1, 4, 7, 10, 6][i]
    sc = scorer(its, Constraints(max_key_distance=0), "plateau")
    done, nearest = beam(sc)
    assert not done
    r = infeasible(sc, sc.cons, nearest)
    assert "max_key_distance" in r["infeasible"] and r["nearest"]


def test_deterministic_and_high_effort_never_worse():
    its = items(12, seed=7)
    sc = scorer(its, arc="wave", target=2400.0)
    done, _ = beam(sc)
    assert done == beam(sc)[0]
    base = sc.total(improve(sc, done[0][1]))
    hi = [improve(sc, done[0][1], "high", s) for s in (0, 1)]
    assert hi[0] == improve(sc, done[0][1], "high", 0)
    assert all(sc.total(h) <= base + 1e-9 for h in hi)


def test_fit_hits_target():
    plays = fit([[32, 40, 48, 56, 64]] * 3, [120.0] * 3, 304.0)
    assert abs(sum(p * 2 for p in plays) - 304) <= 2


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("sb")
    for n, (bpm, hz) in {"a": (124, 55), "b": (126, 62), "c": (124, 58), "d": (128, 65), "e": (90, 70)}.items():
        make_track(d / f"{n}.wav", bpm=bpm, bars=64, bass_hz=hz)
    lib = Library(d)
    for n in "abcde":
        lib.analyze(f"{n}.wav")
    return lib


def run(lib, **kw):
    args = dict(pool=None, duration_min=3.0, energy_arc="build", tempo_arc=None, key_arc=None, tempo_follows_energy=False, key_follows_energy=False,
                constraints={"max_bpm_jump_percent": 8}, effort="normal", seed=0, weights=None, goals=["smooth"], include_plan=True)
    args.update(kw)
    return SB.build(lib, **args)


def test_build_produces_valid_trimmed_plan(lib):
    r = run(lib)
    assert r["infeasible"] is None and r["plan_errors"] == [] and r["audit"]["errors"] == []
    assert P.validate_plan(r["plan"], lib) == []
    assert abs(r["duration_sec"] - 180) <= 180 * 0.15
    assert all(o["src_start_bar"] == int(o["src_start_bar"]) and o["bars"] == int(o["bars"]) for o in r["order"])
    assert len(r["alternatives"]) >= 1 and r["cost"]["total"] >= 0
    assert r["n_tracks"] == len(r["plan"]["clips"]) == len(r["order"])


def test_build_is_deterministic_and_reports_infeasible(lib):
    a, b = run(lib), run(lib)
    assert [o["id"] for o in a["order"]] == [o["id"] for o in b["order"]]
    assert [o["bars"] for o in a["order"]] == [o["bars"] for o in b["order"]]
    bad = run(lib, constraints={"max_key_distance": 0, "max_bpm_jump_percent": 0.0001})
    assert bad["infeasible"] and "nearest" in bad


def test_build_constraints_pool_history_and_explain(lib):
    r = run(lib, constraints={"start_with": "c.wav", "must_exclude": ["e.wav"], "max_bpm_jump_percent": 8})
    assert r["order"][0]["file"] == "c.wav" and all(o["file"] != "e.wav" for o in r["order"])
    lib.store.save_crate("fast", {"all": [{"field": "bpm", "op": "gte", "value": 120}]})
    r = run(lib, pool="fast")
    assert all(o["file"] != "e.wav" for o in r["order"])
    first = r["order"][0]["id"]
    lib.store.history.append("old", [{"id": first, "start_sec": 0, "end_sec": 10}])
    r = run(lib, pool="fast", constraints={"avoid_played_within_days": 1, "max_bpm_jump_percent": 8})
    assert all(o["id"] != first for o in r["order"])
    model, errs = P.build_plan(r["plan"], lib)
    assert errs == []
    ex = SB.explain(lib, model, "build", None, None, False, False, None, None)
    assert ex["violations"] == [] and len(ex["order"]) == r["n_tracks"] and ex["cost"]["total"] >= 0
    with pytest.raises(OpenMixingError):
        run(lib, constraints={"bogus": 1})


def test_clamp_to_clip_cuts_automation_at_the_clip_end():
    from openmixing.mixing import clamp_to_clip
    from openmixing.model.clip import Clip
    c = Clip("c", "t", 0.0, 16.0)
    c.lanes["volume"].points = [[0.0, 1.0], [15.0, 1.0], [18.0, 0.0]]
    clamp_to_clip(c)
    assert c.lanes["volume"].points[-1][0] == 16.0
    assert 0.0 < c.lanes["volume"].points[-1][1] < 1.0
