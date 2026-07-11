#!/usr/bin/env python3
"""Do RA / SCI / C2-contrast actually measure what the paper says they measure?

Three implementations of C2 contrast exist in this repo and they disagree in
sign; the one that produced the published CSV was deleted. Before any of these
numbers go into a paper, the metrics need to be run against population codes
whose structure we *constructed*, so we know the right answer.

Four ground-truth codes on the 18x18 arena, each 324 positions x N units:

  disambiguated  every position has its own smooth code           (no collapse)
  c2_collapse    h[p] == h[R180 p]; 90-deg partners stay distinct (C2 subgroup)
  c4_collapse    h[p] == h[R90 p] for all four images             (full C4 fold)
  random         i.i.d noise                                      (no structure)

Expected, and what the test asserts:
  * RA (90-deg rotational autocorrelation) is high ONLY under c4_collapse.
    It is near zero under c2_collapse -- a C2-symmetric code is NOT invariant
    under 90 degrees. This is why "RA scales monotonically with symmetry order"
    overstates: RA is a 90-deg statistic, so s2 has no reason to score high.
  * SCI (mean symmetry-pair distance / mean random-pair distance, LOWER = more
    collapse) drops toward 0 under the collapse that matches its pair set.
  * C2 contrast separates c2_collapse from the rest, and -- critically -- is
    near zero for BOTH `disambiguated` and `c4_collapse`. A single C2 number
    therefore cannot distinguish "no collapse" from "total C4 collapse"; it must
    be read together with SCI. That ambiguity is the likely source of the
    sign confusion between the three implementations.

Run:  python -m project5_symmetry.analysis.validate_metrics_synthetic
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr

ARENA = 18
N_UNITS = 120
RNG = np.random.default_rng(0)


# ── grid helpers ─────────────────────────────────────────────────────────────

def positions():
    """(324, 2) as (col, row), 1-indexed -- the layout of evaluation.pkl."""
    return np.array([(c, r) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1)],
                    dtype=int)


def rot90(pos):
    """(col,row) -> 90 deg CW on a 1-indexed ARENA x ARENA grid."""
    c, r = pos[:, 0], pos[:, 1]
    return np.stack([ARENA + 1 - r, c], axis=1)


def canonical_index(pos):
    return (pos[:, 1] - 1) * ARENA + (pos[:, 0] - 1)


def orbit_label(pos, order):
    """Smallest canonical index in each position's rotation orbit."""
    cur, best = pos.copy(), canonical_index(pos)
    for _ in range(order - 1):
        cur = rot90(cur) if order == 4 else rot90(rot90(cur))
        best = np.minimum(best, canonical_index(cur))
    return best


# ── the four ground-truth codes ──────────────────────────────────────────────

def make_code(kind, pos):
    n = len(pos)
    if kind == 'random':
        return RNG.normal(size=(n, N_UNITS))

    # smooth place-like basis: each unit is a Gaussian bump at a random centre
    centres = RNG.uniform(1, ARENA, size=(N_UNITS, 2))
    if kind == 'disambiguated':
        key = pos.astype(float)
    elif kind == 'c2_collapse':
        lab = orbit_label(pos, 2)
        key = np.stack([lab % ARENA, lab // ARENA], axis=1).astype(float) + 1
    elif kind == 'c4_collapse':
        lab = orbit_label(pos, 4)
        key = np.stack([lab % ARENA, lab // ARENA], axis=1).astype(float) + 1
    else:
        raise ValueError(kind)
    d2 = ((key[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * 3.0 ** 2))


# ── the metrics, as the codebase defines them ────────────────────────────────

def tuning_maps(h, pos):
    maps = np.full((h.shape[1], ARENA, ARENA), np.nan)
    maps[:, pos[:, 1] - 1, pos[:, 0] - 1] = h.T
    return maps


def ra(h, pos):
    """Mean Pearson r between each unit's rate map and its 90-deg rotation."""
    out = []
    for m in tuning_maps(h, pos):
        a, b = np.nan_to_num(m).ravel(), np.nan_to_num(np.rot90(m)).ravel()
        if a.std() > 1e-9 and b.std() > 1e-9:
            out.append(pearsonr(a, b)[0])
    return float(np.mean(out)) if out else np.nan


def sym_pairs(pos, order):
    idx = {tuple(p): i for i, p in enumerate(pos)}
    pairs = set()
    for i, p in enumerate(pos):
        q = p[None, :]
        for _ in range(order - 1):
            q = rot90(q) if order == 4 else rot90(rot90(q))
            j = idx.get(tuple(q[0]))
            if j is not None and j != i:
                pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def sci(h, pos, order):
    """mean(sym-pair distance) / mean(random-pair distance). LOWER = more collapse."""
    hn = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-8, None)
    pairs = sym_pairs(pos, order)
    if not pairs:
        return np.nan
    sym = np.array([np.linalg.norm(hn[a] - hn[b]) for a, b in pairs])
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(hn), 5000), rng.integers(0, len(hn), 5000)
    keep = a != b
    rand = np.linalg.norm(hn[a[keep]] - hn[b[keep]], axis=1)
    return float(sym.mean() / rand.mean())


def c2_contrast_distance(h, pos):
    """mean(D[C2 pairs]) - mean(D[C4 pairs]); NEGATIVE => 180-deg partners closer.

    This is full_analysis_part1.py's convention, and the one r.tex's prose defines.
    """
    D = squareform(pdist(h - h.mean(0), 'euclidean'))
    p2 = sym_pairs(pos, 2)
    p4 = [p for p in sym_pairs(pos, 4) if p not in set(p2)]
    if not p2 or not p4:
        return np.nan
    return float(np.mean([D[i, j] for i, j in p2]) - np.mean([D[i, j] for i, j in p4]))


# ── report ───────────────────────────────────────────────────────────────────

def main():
    pos = positions()
    kinds = ['disambiguated', 'c2_collapse', 'c4_collapse', 'random']
    rows = {}
    for k in kinds:
        h = make_code(k, pos)
        rows[k] = dict(RA=ra(h, pos), SCI_c2=sci(h, pos, 2), SCI_c4=sci(h, pos, 4),
                       C2=c2_contrast_distance(h, pos))

    print('Synthetic ground truth: 324 positions x %d units, arena %dx%d\n' % (N_UNITS, ARENA, ARENA))
    print(f"{'code':>16}{'RA(90deg)':>12}{'SCI(C2 pairs)':>15}{'SCI(C4 pairs)':>15}{'C2 contrast':>14}")
    for k in kinds:
        r = rows[k]
        print(f"{k:>16}{r['RA']:>12.3f}{r['SCI_c2']:>15.3f}{r['SCI_c4']:>15.3f}{r['C2']:>14.3f}")

    print('\nchecks')
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    chk('RA is high only under C4 collapse',
        rows['c4_collapse']['RA'] > 0.8 and rows['c2_collapse']['RA'] < 0.4)
    chk('RA does NOT detect C2 collapse (it is a 90-deg statistic)',
        rows['c2_collapse']['RA'] < rows['c4_collapse']['RA'] / 2)
    chk('SCI(C2 pairs) collapses under C2 code',
        rows['c2_collapse']['SCI_c2'] < 0.1 < rows['disambiguated']['SCI_c2'])
    chk('SCI(C4 pairs) collapses under C4 code',
        rows['c4_collapse']['SCI_c4'] < 0.1 < rows['disambiguated']['SCI_c4'])
    chk('C2 contrast is strongly negative ONLY under C2 collapse',
        rows['c2_collapse']['C2'] < -0.05 and rows['c4_collapse']['C2'] > -0.05)
    # Scale-free: both must be a small fraction of the C2-collapse signal. An
    # absolute cutoff would just encode the arbitrary units of the code.
    ref = abs(rows['c2_collapse']['C2'])
    chk('C2 contrast CANNOT separate no-collapse from full-C4 collapse '
        '(both <10% of the C2 signal) -- must be read with SCI',
        abs(rows['disambiguated']['C2']) < 0.1 * ref
        and abs(rows['c4_collapse']['C2']) < 0.1 * ref)
    chk('full C4 collapse also collapses the C2 pair set (C4 implies C2)',
        rows['c4_collapse']['SCI_c2'] < 0.1)
    chk('C2 collapse does NOT collapse the C4 pair set',
        rows['c2_collapse']['SCI_c4'] > 0.5)

    print('\n' + ('ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
