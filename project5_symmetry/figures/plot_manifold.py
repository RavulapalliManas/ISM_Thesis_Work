#!/usr/bin/env python3
"""Draw the population manifold, coloured by C2 phase, for one checkpoint per panel.

The point cloud is the POSITION-CONDITIONED MEAN population vector: one point per arena
cell, h(x) = E[h | agent at x]. That is the embedding of the arena into neural space,
and it is the object whose topology should mirror the arena's. Plotting raw per-timestep
hidden states instead buries the structure in heading- and phase-of-trajectory noise
(PC1-3 explain only ~32% of that cloud, versus ~80% of this one).

Under folding onto X/G the map obeys h(x) == h(R^2 x), so the 324 cells collapse pairwise
onto 162 points and the two phase colours superimpose exactly. `fold_ratio` quantifies it:
the distance between a cell and its group image, in units of the typical distance to a
spatial neighbour. Folded => ~0. Unfolded => ~1 or more.

PCA only. UMAP is fine for a prettier picture but must never be used to *measure*
topology -- it can create and destroy loops. Gardner et al. (2022) likewise used UMAP
for visualisation and computed persistent cohomology on a 6-D PCA space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_phase_decoding import orbit_and_phase
from project5_symmetry.analysis.run_spectrum import (collect, identify,
                                                     model_from_checkpoint)
from project5_symmetry.training.dataset import TrajectoryDataset


def position_means(hidden, pos, min_count=8):
    """h(x) = E[h | agent at x]. Returns (cells, H) and the matching (cells, 2) coords."""
    keys, inv, counts = np.unique(pos, axis=0, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    means = np.stack([hidden[inv == i].mean(0) for i in range(len(keys))])
    return means[keep], keys[keep]


def fold_ratio(means, cells):
    """||h(x) - h(R^2 x)|| relative to the typical distance to a spatial neighbour."""
    from project5_symmetry.analysis.run_phase_decoding import rot180
    index = {tuple(c): i for i, c in enumerate(cells)}
    pair, nbr = [], []
    for i, c in enumerate(cells):
        j = index.get(tuple(rot180(c[None])[0]))
        if j is not None and j != i:
            pair.append(np.linalg.norm(means[i] - means[j]))
        k = index.get((c[0] + 1, c[1]))
        if k is not None:
            nbr.append(np.linalg.norm(means[i] - means[k]))
    return float(np.mean(pair) / np.mean(nbr))


def panel(ax, H3, colour, title, cmap, cbar_label=None):
    s = ax.scatter(H3[:, 0], H3[:, 1], H3[:, 2], c=colour, s=16, alpha=.85,
                   cmap=cmap, linewidths=0)
    ax.set_title(title, fontsize=10)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.set_xlabel('PC1', fontsize=7); ax.set_ylabel('PC2', fontsize=7)
    ax.set_zlabel('PC3', fontsize=7)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-states', type=int, default=6000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(4)
    dev = torch.device('cpu')
    n = len(a.ckpts)
    fig = plt.figure(figsize=(5.2 * n, 9.6))
    datasets = {}

    for j, cp in enumerate(a.ckpts):
        p = Path(cp)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        hidden, pos = collect(model, datasets[cond], hd, a.n_states, dev)
        means, cells = position_means(hidden, pos)
        _, phase = orbit_and_phase(cells, 'c2')
        pca = PCA(n_components=3).fit(means)
        H3 = pca.transform(means)
        fr = fold_ratio(means, cells)

        ax = fig.add_subplot(2, n, j + 1, projection='3d')
        panel(ax, H3, phase, f'{cond} / {hd}\nC2 phase   fold ratio = {fr:.2f}', 'coolwarm')
        ax = fig.add_subplot(2, n, n + j + 1, projection='3d')
        panel(ax, H3, cells[:, 0], f'{cond} / {hd}\nx position', 'viridis')
        print(f'  {cond}/{hd}: {len(cells)} cells, PC1-3 explain '
              f'{pca.explained_variance_ratio_.sum():.1%}, fold_ratio={fr:.3f}', flush=True)

    fig.suptitle('Position-conditioned population manifold, PCA(3). One point per arena cell.\n'
                 'Top: C2 phase. Folding superimposes red and blue (fold ratio -> 0).', fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
