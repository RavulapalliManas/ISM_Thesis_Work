"""Pins the Proposition: RA is the (k=0 minus k=2) isotypic power under C4.

If this ever fails, either the RA definition or the character convention moved,
and every conclusion drawn from the spectrum needs re-deriving.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from project5_symmetry.analysis.isotypic import (isotypic_power, odd_power,
                                                 ra_from_spectrum)

RNG = np.random.default_rng(0)
ARENA = 18


def _ra_pearson(f):
    a, b = f.ravel(), np.rot90(f).ravel()
    return pearsonr(a, b)[0]


def _c4_folded():
    """f(x) == f(R x): constant on every C4 orbit."""
    base = RNG.normal(size=(ARENA, ARENA))
    return sum(np.rot90(base, j) for j in range(4))


def _c2_folded():
    """f(x) == f(R^2 x) but not C4-symmetric."""
    base = RNG.normal(size=(ARENA, ARENA))
    return base + np.rot90(base, 2)


def test_parseval_power_sums_to_norm():
    f = RNG.normal(size=(ARENA, ARENA))
    fc = f - f.mean()
    assert isotypic_power(f).sum() == pytest.approx((fc ** 2).sum(), rel=1e-10)


def test_conjugate_characters_have_equal_power_for_real_maps():
    for f in (RNG.normal(size=(ARENA, ARENA)), _c2_folded(), _c4_folded()):
        P = isotypic_power(f)
        assert P[1] == pytest.approx(P[3], abs=1e-9)


@pytest.mark.parametrize('make', [
    lambda: RNG.normal(size=(ARENA, ARENA)),
    _c2_folded,
    _c4_folded,
])
def test_ra_equals_p0_minus_p2(make):
    f = make()
    assert _ra_pearson(f) == pytest.approx(ra_from_spectrum(isotypic_power(f)), abs=1e-9)


def test_ra_is_blind_to_c2_structure():
    """A perfectly C2-folded code reads RA ~ 0 -- the same as noise."""
    P = isotypic_power(_c2_folded())
    assert odd_power(P) == pytest.approx(0.0, abs=1e-9)      # all power is even
    assert abs(ra_from_spectrum(P)) < 0.35                   # yet RA is uninformative


def test_ra_is_one_for_c4_folded_code():
    P = isotypic_power(_c4_folded())
    assert ra_from_spectrum(P) == pytest.approx(1.0, abs=1e-9)
    assert P[1] == pytest.approx(0.0, abs=1e-9)
    assert P[2] == pytest.approx(0.0, abs=1e-9)
