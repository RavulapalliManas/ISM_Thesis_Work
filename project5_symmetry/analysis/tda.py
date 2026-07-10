"""Persistent homology of a neural point cloud, following Gardner et al. (2022) and
Hermansen, Klindt & Dunn (2024).

Pipeline (theirs, as published):
    PCA-whiten to 6-8 dimensions
    two-step density downsampling to ~1200-2200 points   (Rips persistence is polynomial)
    ripser, cosine metric, Z_47 coefficients
    read Betti numbers off the barcode: a torus is one H0 bar, two H1 bars, one H2 bar

The Betti number is estimated from the largest gap in the sorted bar lifetimes, but only
among bars that clear a NOISE FLOOR obtained by shuffling. Both are needed. A gap rule on
its own is purely multiplicative and therefore scale-free, so it can say "which bars are
longer than the others" but never "none of these is a loop": `argmax(ratios) + 1 >= 1`
always. Fed a disk, it answers b1 = 1. That is not a hypothetical -- the `open` arena
(true b1 = 0) was scored wrong at every checkpoint of every seed for exactly this reason,
and the b1 = 0 unit tests had been written to bypass `betti1` and inspect raw lifetimes,
so nothing caught it.

The floor supplies the missing absolute scale: the longest H1 bar a cloud with the same
per-coordinate marginals but no joint structure produces. Gardner et al. and Hermansen,
Klindt & Dunn both threshold against such a null. `betti_from_gap(floor=0)` retains the
old scale-free behaviour and is kept only so the gap logic can be tested in isolation.

UMAP is never used here. It can create and destroy loops; Gardner et al. likewise used it
only to draw pictures, and computed persistence on the PCA space.
"""
from __future__ import annotations

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

COEFF = 47          # Z_47, as in Hermansen & Dunn
N_PCS = 6
N_POINTS = 1200


def _density_subsample(X, n_points, k=15, seed=0):
    """Keep the highest-density points, then thin them radially so the sample is spread.

    Density = -mean distance to the k nearest neighbours.
    """
    rng = np.random.default_rng(seed)
    if len(X) <= n_points:
        return X
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, _ = tree.query(X, k=min(k + 1, len(X)))
    density = -d[:, 1:].mean(1)
    keep = np.argsort(density)[-min(len(X), 3 * n_points):]
    rng.shuffle(keep)
    return X[keep[:n_points]]


def prepare(X, n_pcs=N_PCS, n_points=N_POINTS, whiten=True, seed=0):
    """Neural point cloud -> low-dimensional, subsampled cloud ready for ripser."""
    X = np.asarray(X, dtype=float)
    n_pcs = min(n_pcs, X.shape[1], X.shape[0])
    Y = PCA(n_components=n_pcs, whiten=whiten, random_state=seed).fit_transform(X)
    return _density_subsample(Y, n_points, seed=seed)


def lifetimes(X, maxdim=1, metric='euclidean', coeff=COEFF):
    """Sorted (descending) finite bar lifetimes per homology dimension.

    `maxdim=1` by default, and that is not a detail. Measured on this machine:

        maxdim=1:  n=400 -> 0.06 s,  n=1600 -> 3.2 s
        maxdim=2:  n=300 -> 0.94 s,  n=1200 -> does not finish

    H2 costs roughly n^4 and is only needed to certify a torus. Our arenas are planar,
    so b2 = 0 everywhere and H1 is the whole signal. Ask for maxdim=2 only on a few
    hundred points.
    """
    dgms = ripser(np.asarray(X, dtype=float), maxdim=maxdim, coeff=coeff, metric=metric)['dgms']
    out = []
    for d in dgms:
        if len(d) == 0:
            out.append(np.array([]))
            continue
        finite = d[np.isfinite(d[:, 1])]
        lt = finite[:, 1] - finite[:, 0]
        out.append(np.sort(lt)[::-1])
    return out


def noise_floor(X, metric='euclidean', coeff=COEFF, n_shuffles=32, seed=0, percentile=99.0):
    """Longest H1 bar produced by a cloud with X's marginals and none of its structure.

    Each coordinate is permuted independently across points: every column keeps its own
    distribution, the joint geometry is destroyed. A real loop must outlive this.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    tops = []
    for _ in range(n_shuffles):
        Z = np.column_stack([rng.permutation(col) for col in X.T])
        lt = lifetimes(Z, maxdim=1, metric=metric, coeff=coeff)[1]
        tops.append(float(lt[0]) if len(lt) else 0.0)
    return float(np.percentile(tops, percentile))


def betti_from_gap(lt, max_betti=4, floor=0.0):
    """(betti, confidence) from the largest multiplicative gap among bars above `floor`.

    `floor = 0` is the scale-free rule and CANNOT return a positive-length barcode's b = 0;
    it survives only for unit-testing the gap logic. With a floor:

        lt[0] <= floor          -> (0, floor / lt[0]),  confidence > 1 means clearly no loop
        otherwise               -> gap over the surviving bars, terminated by the floor, so
                                   the last real bar can win the gap against it

    confidence = lt[b-1] / lt[b], how far the last accepted bar stands above the first
    rejected one. 1.0 means no separation at all.
    """
    lt = np.asarray(lt, dtype=float)
    lt = np.sort(lt[lt > 0])[::-1]
    if len(lt) == 0:
        return 0, float('inf')
    if floor > 0:
        if lt[0] <= floor:
            return 0, float(floor / max(lt[0], 1e-12))
        n_sig = min(int((lt > floor).sum()), max_betti)
        # Terminate at the floor (or the first sub-floor bar) so a barcode whose every
        # significant bar is real -- [10, 9] above a floor of 1 -- reads b = 2, not b = 1.
        term = max(floor, lt[n_sig]) if n_sig < len(lt) else floor
        cand = np.concatenate([lt[:n_sig], [term]])
    else:
        cand = lt[:max_betti + 1]
    if len(cand) == 1:
        return 1, float('inf')
    ratios = cand[:-1] / np.maximum(cand[1:], 1e-12)
    b = int(np.argmax(ratios)) + 1
    return b, float(ratios[b - 1])


def betti1(X, metric='euclidean', n_shuffles=32, seed=0, percentile=99.0, max_betti=4, **kw):
    """Estimated b1 of a point cloud, with its confidence.

    `n_shuffles = 0` disables the null and restores the old scale-free rule, which reads
    b1 >= 1 on every non-degenerate cloud. Do not use it to score a b1 = 0 arena.
    """
    lt = lifetimes(X, maxdim=1, metric=metric, **kw)[1]
    floor = (noise_floor(X, metric=metric, n_shuffles=n_shuffles, seed=seed,
                         percentile=percentile, **kw) if n_shuffles else 0.0)
    return betti_from_gap(lt, max_betti=max_betti, floor=floor)


def h1_gap_score(X, expected_b1, metric='euclidean', **kw):
    """How well the barcode supports exactly `expected_b1` loops.

    Returns the ratio lifetime[expected_b1 - 1] / lifetime[expected_b1] (inf if there is
    no (b1+1)-th bar), and 0.0 when expected_b1 == 0 and any bar is long-lived.
    For b1 == 0 the score is 1 / (longest H1 lifetime), large when there is no loop.
    """
    lt = lifetimes(X, maxdim=1, metric=metric, **kw)[1]
    if expected_b1 == 0:
        return float('inf') if len(lt) == 0 else 1.0 / max(lt[0], 1e-12)
    if len(lt) < expected_b1:
        return 0.0
    if len(lt) == expected_b1:
        return float('inf')
    return float(lt[expected_b1 - 1] / max(lt[expected_b1], 1e-12))
