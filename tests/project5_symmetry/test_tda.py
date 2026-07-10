"""The TDA readout must recover known Betti numbers, including on the real arena cells.

If it cannot see a loop in a literal annulus of grid cells, it will not see one in the
neural manifold either, and a null result would be meaningless.
"""
from __future__ import annotations

import numpy as np
import pytest

from project5_symmetry.analysis.tda import betti1, betti_from_gap, lifetimes, noise_floor
from project5_symmetry.environments.topology_arenas import (ARENA, EXPECTED_B1, LAYOUTS,
                                                            is_passable)

# Each fixture draws from its OWN generator. A module-level RNG consumed in draw order makes
# every cloud depend on how many tests ran before it: adding one test silently changes the
# torus, and a barcode confidence drifts across an unrelated edit.
def _circle(n=400, r=1.0, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))


def _disk(n=400, seed=2):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _torus(n=300, R=2.0, r=1.0, seed=3):   # maxdim=2 is ~n^4; 300 costs ~1 s, 1200 never finishes
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    return np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u),
                     r * np.sin(v)], 1)


def _cells(layout):
    return np.array([(r, c) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1)
                     if is_passable(layout, r, c)], dtype=float)


def test_circle_has_one_loop():
    b, conf = betti1(_circle(), seed=0)
    assert b == 1
    # `conf` now means "the last accepted bar as a multiple of the shuffle noise floor",
    # not "as a multiple of the next bar". For this circle those are 4.8x and 70x. The
    # floor ratio is the smaller, stricter number, and the only one that can refute a
    # spurious loop, so it is the one reported.
    assert conf > 3, f'circle loop is only {conf:.2f}x the noise floor'


def test_disk_has_no_loop():
    lt = lifetimes(_disk(), maxdim=1)[1]
    longest = lt[0] if len(lt) else 0.0
    circ = lifetimes(_circle(), maxdim=1)[1][0]
    assert longest < 0.25 * circ, 'a disk produced a loop as long-lived as a circle'


def test_betti1_reports_zero_for_a_disk():
    """The test the old suite would not write. `test_disk_has_no_loop` inspects raw lifetimes
    and never calls `betti1`, because `betti1` answered 1. A scale-free gap rule computes
    `argmax(ratios) + 1 >= 1`, so b = 0 was unreachable and the `open` arena (true b1 = 0)
    was scored wrong at every checkpoint."""
    b, conf = betti1(_disk(), seed=0)
    assert b == 0, f'a disk read as b1={b}'
    assert conf > 1.0, 'confidence should say the top bar sits below the noise floor'


def test_the_scale_free_rule_still_cannot_see_an_empty_barcode():
    """Pins the bug so it cannot come back silently: with the null disabled, a disk reads 1."""
    b, _ = betti1(_disk(), n_shuffles=0)
    assert b >= 1, 'n_shuffles=0 is the old scale-free rule and must be documented as broken'


def test_the_null_separates_a_circle_from_a_disk():
    """The claim without a magic threshold: the circle's longest bar towers over its own
    noise floor, the disk's does not even reach its own."""
    c_lt, c_f = lifetimes(_circle(), maxdim=1)[1][0], noise_floor(_circle(), seed=0)
    d_lt, d_f = lifetimes(_disk(), maxdim=1)[1][0], noise_floor(_disk(), seed=0)
    assert c_lt / c_f > 1.0, 'a circle must clear its noise floor'
    assert d_lt / d_f < 1.0, 'a disk must not clear its noise floor'
    assert (c_lt / c_f) > 5 * (d_lt / d_f)


def test_torus_barcode_is_one_two_one():
    """The Gardner et al. signature: one H0 bar, two H1 bars, one H2 bar.
    Included to prove the pipeline *could* see a torus -- our arenas are planar (b2=0),
    and this network has no grid cells, so no torus is expected in the real data."""
    lt = lifetimes(_torus(), maxdim=2)
    b1, c1 = betti_from_gap(lt[1])
    b2, _ = betti_from_gap(lt[2])
    assert b1 == 2, f'torus H1 read as {b1}'
    assert b2 == 1, f'torus H2 read as {b2}'
    assert c1 > 1.5


def test_betti_from_gap_picks_the_step():
    assert betti_from_gap(np.array([10.0, 9.0, 0.4, 0.3, 0.2]))[0] == 2
    assert betti_from_gap(np.array([10.0, 0.4, 0.3]))[0] == 1
    assert betti_from_gap(np.array([]))[0] == 0


def test_betti_from_gap_with_a_floor_can_return_zero():
    # every bar below the floor -> no loop at all
    assert betti_from_gap(np.array([0.4, 0.3]), floor=1.0)[0] == 0
    # the floor terminates the candidate list, so two clearly-real bars read as 2, not 1
    assert betti_from_gap(np.array([10.0, 9.0, 0.4, 0.3]), floor=1.0)[0] == 2
    assert betti_from_gap(np.array([10.0, 0.4, 0.3]), floor=1.0)[0] == 1


def test_betti_from_gap_floor_zero_is_the_legacy_scale_free_rule():
    """Kept only for unit-testing the gap. It cannot express 'no loop'."""
    assert betti_from_gap(np.array([0.4, 0.3]), floor=0.0)[0] >= 1


@pytest.mark.parametrize('layout', LAYOUTS)
def test_arena_cells_have_the_arenas_betti_number(layout):
    """The ground truth: the passable cells, as raw (row, col) points.

    Every layout is asserted through the SAME estimator, `open` included. The previous
    version special-cased `exp == 0` to inspect raw lifetimes, which is precisely the case
    the estimator got wrong.
    """
    b, conf = betti1(_cells(layout), seed=0)
    exp = EXPECTED_B1[layout]
    assert b == exp, f'{layout}: TDA read b1={b} (conf {conf:.2f}), truth {exp}'
