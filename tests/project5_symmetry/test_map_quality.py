"""Map quality and folding are different questions. Pin the trap.

The dangerous mistake is to read a high cross-seed RSA as evidence that the code did NOT fold.
Two seeds that both fold onto X/C2 have beautifully consistent representational geometry --
the folding is reproducible. Cross-seed RSA answers "is this a map?", never "is it identifiable?"

Spatial RSA is a different beast: it correlates neural distance with *spatial* distance, so a
folded pair (zero neural distance, large spatial distance) drags it down. It is sensitive to
folding, though not diagnostic of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from project5_symmetry.analysis.run_map_quality import position_rdm
from project5_symmetry.evaluation.metrics import cross_seed_rsa_alignment, srsa

ARENA = 18
RNG = np.random.default_rng(0)


def _cells():
    return np.array([(r, c) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1)], float)


def _rot180(p):
    return (ARENA + 1) - p


def _place_code(pos, seed, folded, n_units=200, sigma=2.0):
    """Gaussian place code. If folded, h(x) == h(R^2 x) by construction."""
    rng = np.random.default_rng(seed)
    X = _rot180(pos) if not folded else np.minimum(pos, _rot180(pos))
    if folded:
        # canonical representative of the C2 orbit -> identical code at x and R^2 x
        canon = np.where((pos[:, 0] < _rot180(pos)[:, 0])[:, None], pos, _rot180(pos))
        X = canon
    else:
        X = pos
    C = X[rng.integers(0, len(X), n_units)]
    D2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return np.exp(-D2 / (2 * sigma ** 2))


def test_a_folded_code_scores_high_cross_seed_rsa():
    """THE TRAP. Two seeds that both fold agree with each other almost perfectly."""
    pos = _cells()
    mats = []
    for seed in (1, 2, 3):
        h = _place_code(pos, seed, folded=True)
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        mats.append(1.0 - hn @ hn.T)
    rho = cross_seed_rsa_alignment(mats)['mean_rho']
    assert rho > 0.9, f'folded seeds should agree; got cross-seed rho = {rho}'


def test_unfolded_seeds_also_agree_so_cross_seed_rsa_cannot_separate_them():
    pos = _cells()
    mats = [ ]
    for seed in (1, 2, 3):
        h = _place_code(pos, seed, folded=False)
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        mats.append(1.0 - hn @ hn.T)
    rho = cross_seed_rsa_alignment(mats)['mean_rho']
    assert rho > 0.9
    # both regimes score high -> the measure is blind to folding, which is the whole point
    folded = []
    for seed in (1, 2, 3):
        h = _place_code(pos, seed, folded=True)
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        folded.append(1.0 - hn @ hn.T)
    assert cross_seed_rsa_alignment(folded)['mean_rho'] > 0.9


def test_spatial_rsa_IS_dragged_down_by_folding():
    """Neural-vs-space RSA sees the fold: orbit-mates are neurally identical but spatially far."""
    pos = _cells()
    unfolded = _place_code(pos, 1, folded=False)
    folded = _place_code(pos, 1, folded=True)
    su = srsa(unfolded, pos, space_metric='euclidean', max_n=len(pos))
    sf = srsa(folded, pos, space_metric='euclidean', max_n=len(pos))
    assert su > sf + 0.15, f'folding should lower spatial RSA: unfolded {su:.3f}, folded {sf:.3f}'
    assert su > 0.7


def test_a_dead_code_scores_low_on_both():
    pos = _cells()
    mats, noise = [], None
    for seed in (1, 2, 3):
        noise = RNG.normal(size=(len(pos), 200))
        n = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12)
        mats.append(1.0 - n @ n.T)
    assert abs(cross_seed_rsa_alignment(mats)['mean_rho']) < 0.2
    assert abs(srsa(noise, pos, space_metric='euclidean', max_n=len(pos))) < 0.2


def test_position_rdm_is_symmetric_with_zero_diagonal():
    pos = _cells()
    h = _place_code(pos, 1, folded=False)
    hidden = np.repeat(h, 2, axis=0)
    positions = np.repeat(pos, 2, axis=0).astype(int)
    D = position_rdm(hidden, positions, ARENA)
    assert D.shape[0] == D.shape[1]
    assert np.allclose(D, D.T, atol=1e-6)
    assert np.allclose(np.diagonal(D), 0, atol=1e-6)
