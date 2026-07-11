"""Isotypic (character) decomposition of a population code under the arena's
rotational symmetry group.

Why this exists
---------------
The paper reports three bespoke scalars -- RA, SCI, C2 contrast -- whose three
surviving implementations disagree in sign, and whose published values came from
a script that no longer exists. They are all shadows of one object.

Rotation by 90 degrees acts on functions over a square arena as the generator of
the cyclic group C4. Any rate map f decomposes orthogonally into character
components (omega = i):

    P_k f = (1/4) sum_j omega^{-jk} (f o R^j),      k = 0,1,2,3

Rotation sends P_k f -> omega^k P_k f, so with `RA` defined as the correlation of
a map with its 90-degree rotation, and Re(i^k) = 1, 0, -1, 0:

    Proposition.   RA = (P0 - P2) / (P0 + P1 + P2 + P3)

verified numerically to machine precision (see tests). Two consequences:

  * RA is analytically BLIND to C2 structure. A perfectly 180-degree-folded code
    has all its power in the even components, and P0 cancels P2. It reads ~0 --
    the same as a code with no structure at all. The paper's "S2 approx S1" null
    is a property of the metric, not of the networks.
  * "RA scales monotonically with symmetry order" is not what RA measures. RA is
    a 90-degree statistic; a C2 arena has no reason to score on it.

The right readout is the full spectrum (P0..P3), which is what this module
computes. On the trained networks it separates the conditions exactly as the
group theory predicts: the C2 arena suppresses odd power and enriches P2; the C4
arena enriches the trivial component P0.

For a real-valued map, P1 = P3 always (conjugate characters), which doubles as an
internal correctness check.
"""
from __future__ import annotations

import numpy as np

ARENA_DEFAULT = 18


def rate_maps(hidden: np.ndarray, positions: np.ndarray, arena: int = ARENA_DEFAULT) -> np.ndarray:
    """(n_pos, n_units) + 1-indexed (col,row) positions -> (n_units, arena, arena)."""
    maps = np.full((hidden.shape[1], arena, arena), np.nan)
    maps[:, positions[:, 1] - 1, positions[:, 0] - 1] = hidden.T
    return np.nan_to_num(maps)


def isotypic_power(field: np.ndarray, remove_constant: bool = True) -> np.ndarray:
    """Power of a single rate map in each C4 character component. Returns (4,).

    Sums to ||field||^2 by Parseval. `remove_constant=True` subtracts the mean, so
    P0 is the non-constant trivial-irrep power -- which is what a Pearson-based RA
    compares against.
    """
    f = field - field.mean() if remove_constant else field
    rots = [np.rot90(f, j) for j in range(4)]
    w = 1j
    return np.array([
        float((np.abs(sum((w ** (-j * k)) * rots[j] for j in range(4)) / 4.0) ** 2).sum())
        for k in range(4)
    ])


def isotypic_spectrum(hidden: np.ndarray, positions: np.ndarray,
                      unit_mask: np.ndarray | None = None,
                      arena: int = ARENA_DEFAULT) -> np.ndarray:
    """Mean normalised power fractions (4,) over the selected units."""
    maps = rate_maps(hidden, positions, arena)
    if unit_mask is not None:
        maps = maps[unit_mask]
    P = np.stack([isotypic_power(f) for f in maps])          # (units, 4)
    total = P.sum(axis=1, keepdims=True)
    good = total[:, 0] > 0
    return (P[good] / total[good]).mean(axis=0)


def ra_from_spectrum(P: np.ndarray) -> float:
    """RA = (P0 - P2) / sum(P). The Proposition, as a one-liner."""
    return float((P[0] - P[2]) / P.sum())


def odd_power(P: np.ndarray) -> float:
    """P1 + P3: the components a C2-symmetric code must suppress."""
    return float(P[1] + P[3])
