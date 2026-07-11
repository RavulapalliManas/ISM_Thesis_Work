"""Phase decoding must read chance for a folded code and ceiling for an unfolded one.

If the orbit/phase construction is wrong, a folded code can still look decodable (or an
unfolded one can look like chance) and the central claim inverts. Pin both ends with
synthetic codes whose ground truth is known.
"""
from __future__ import annotations

import numpy as np
import pytest

from project5_symmetry.analysis.run_phase_decoding import (ARENA, decode,
                                                           orbit_and_phase, rot180)

RNG = np.random.default_rng(0)


def _positions(n=4000):
    return np.stack([RNG.integers(1, ARENA + 1, n), RNG.integers(1, ARENA + 1, n)], 1)


def test_rot180_is_an_involution_with_no_fixed_point():
    p = _positions()
    assert np.array_equal(rot180(rot180(p)), p)
    assert not np.any((rot180(p) == p).all(1))       # ARENA is even


def test_orbit_is_invariant_and_phase_flips_under_the_group():
    p = _positions()
    o1, f1 = orbit_and_phase(p)
    o2, f2 = orbit_and_phase(rot180(p))
    assert np.array_equal(o1, o2)                    # same orbit
    assert np.array_equal(f1, 1 - f2)                # opposite phase


def test_every_orbit_has_exactly_two_cells():
    all_cells = np.array([(x, y) for x in range(1, ARENA + 1) for y in range(1, ARENA + 1)])
    orbit, _ = orbit_and_phase(all_cells)
    counts = np.unique(np.unique(orbit, return_counts=True)[1])
    assert counts.tolist() == [2]


def test_folded_code_reads_chance_and_keeps_the_positive_control():
    """h(x) == h(R^2 x) by construction: phase is unrecoverable, position within the
    fundamental domain is not."""
    pos = _positions(6000)
    canon = np.where((pos[:, 0] < rot180(pos)[:, 0])[:, None], pos, rot180(pos))
    hidden = np.concatenate([canon, canon ** 2, RNG.normal(0, .01, (len(pos), 8))], 1).astype(float)
    acc, raw_r2, dom_r2 = decode(hidden, pos)
    assert acc == pytest.approx(0.5, abs=0.06), f'folded code decoded phase at {acc}'
    assert dom_r2 > 0.9, f'folded code should expose the fundamental domain: {dom_r2}'
    assert raw_r2 < 0.6, f'folded code must NOT expose raw position: {raw_r2}'
    assert max(raw_r2, dom_r2) > 0.9                       # space is still encoded


def test_unfolded_code_decodes_phase_near_ceiling():
    """h(x) carries the raw position, so phase is a linear function of it."""
    pos = _positions(6000)
    hidden = np.concatenate([pos, RNG.normal(0, .01, (len(pos), 8))], 1).astype(float)
    acc, raw_r2, dom_r2 = decode(hidden, pos)
    assert acc > 0.9, f'unfolded code only reached {acc}'
    assert raw_r2 > 0.9, f'unfolded code should expose raw position: {raw_r2}'
    assert dom_r2 < 0.6, 'domain_r2 is a folding indicator, not a positive control'
    assert max(raw_r2, dom_r2) > 0.9


def test_dead_code_is_distinguishable_from_a_folded_one():
    """Chance phase with no spatial signal at all -- must not be read as folding."""
    pos = _positions(4000)
    hidden = RNG.normal(0, 1, (len(pos), 10))
    acc, raw_r2, dom_r2 = decode(hidden, pos)
    assert acc == pytest.approx(0.5, abs=0.06)
    assert max(raw_r2, dom_r2) < 0.2, 'a dead code must fail the spatial sanity check'


# ── C4: the group `const` is invariant under, and the reason s4/const looks "dead" ──

def _c4_folded_code(pos):
    """h(x) == h(Rx) for the quarter turn: constant on every C4 orbit."""
    from project5_symmetry.analysis.run_phase_decoding import canonical
    c = canonical(pos, 'c4').astype(float)
    return np.concatenate([c, c ** 2, RNG.normal(0, .01, (len(pos), 8))], 1)


def test_c4_orbits_and_phases_are_well_formed():
    from project5_symmetry.analysis.run_phase_decoding import orbit_and_phase
    cells = np.array([(x, y) for x in range(1, ARENA + 1) for y in range(1, ARENA + 1)])
    orbit, phase = orbit_and_phase(cells, 'c4')
    assert len(np.unique(orbit)) == 81
    assert np.unique(np.unique(orbit, return_counts=True)[1]).tolist() == [4]
    assert sorted(np.unique(phase)) == [0, 1, 2, 3]


def test_c4_folded_code_reads_chance_under_the_c4_decoder():
    pos = _positions(8000)
    acc, raw_r2, dom_r2 = decode(_c4_folded_code(pos), pos, group='c4')
    assert acc == pytest.approx(0.25, abs=0.06), f'C4-folded code decoded 4-phase at {acc}'
    assert dom_r2 > 0.9, f'C4 fundamental domain must be decodable: {dom_r2}'
    assert raw_r2 < 0.6


def test_c4_folded_code_reproduces_the_s4_const_signature_under_a_c2_decoder():
    """The observed s4/const row: phase ~ 0.5, raw_r2 ~ 0, C2-domain_r2 NEGATIVE.

    A C2 decoder cannot see a C4 fold: even the C2 fundamental-domain coordinate stays
    2-fold ambiguous. This is why s4/const must NOT be read as a dead code.
    """
    pos = _positions(8000)
    acc, raw_r2, dom_r2 = decode(_c4_folded_code(pos), pos, group='c2')
    assert acc == pytest.approx(0.5, abs=0.06)
    assert raw_r2 < 0.3
    assert dom_r2 < 0.3, f'a C2 decoder must fail on a C4-folded code, got {dom_r2}'


def test_c2_folded_code_hits_the_half_ceiling_under_the_c4_decoder():
    """A C2 fold merges the four C4 phases into two pairs, so a C4 decoder tops out at
    exactly 0.5 -- it can name the pair, never the element. This is the ceiling that the
    real s4/axis models sit at (0.482), and it is what separates "folded by C2" from both
    "folded by C4" (0.25) and "not folded" (~0.9)."""
    pos = _positions(8000)
    from project5_symmetry.analysis.run_phase_decoding import canonical
    c = canonical(pos, 'c2').astype(float)
    hidden = np.concatenate([c, c ** 2, RNG.normal(0, .01, (len(pos), 8))], 1)
    acc, _, _ = decode(hidden, pos, group='c4')
    assert acc == pytest.approx(0.5, abs=0.06), f'expected the 1/2 ceiling, got {acc}'
