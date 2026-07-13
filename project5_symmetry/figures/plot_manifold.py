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
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from sklearn.decomposition import PCA

# Match analysis/make_paper_figures.py's house style so this figure sits consistently
# alongside the rest of the paper's plots.
_PREFERRED = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'TeX Gyre Heros', 'DejaVu Sans']
_have = {f.name for f in font_manager.fontManager.ttflist}
_family = next((f for f in _PREFERRED if f in _have and f != 'Helvetica'), 'DejaVu Sans')
plt.rcParams.update({
    'font.size': 7, 'font.family': _family, 'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#222222', 'axes.labelcolor': '#222222', 'text.color': '#222222',
    'xtick.color': '#222222', 'ytick.color': '#222222', 'axes.titlecolor': '#222222',
    'axes.titlesize': 7.5, 'axes.titleweight': 'normal', 'axes.titlepad': 4,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

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
    ax.set_title(title, fontsize=8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.set_xlabel('PC1', fontsize=6.5); ax.set_ylabel('PC2', fontsize=6.5)
    ax.set_zlabel('PC3', fontsize=6.5)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-states', type=int, default=6000)
    ap.add_argument('--n-scatter', type=int, default=2,
                    help='how many conditions get a 3D scatter panel (the most- and '
                         'least-folded, by default); the rest appear only in the fold-ratio bars')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(4)
    dev = torch.device('cpu')
    datasets = {}
    results = []

    for cp in a.ckpts:
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
        results.append(dict(cond=cond, hd=hd, means=means, cells=cells, phase=phase,
                             H3=H3, fold_ratio=fr, evr=pca.explained_variance_ratio_.sum()))
        print(f'  {cond}/{hd}: {len(cells)} cells, PC1-3 explain {results[-1]["evr"]:.1%}, '
              f'fold_ratio={fr:.3f}', flush=True)

    # The argument this figure makes: fold ratio (distance to the group image, in units of
    # distance to a spatial neighbour) separates folded (<1) from unfolded (>1) codes. That
    # single number, not the scatter shape, is the load-bearing quantity, so it leads and gets
    # the largest panel; the scatters are illustrative, not the evidence.
    order = sorted(results, key=lambda r: r['fold_ratio'])
    labels = [f'{r["cond"]}/{r["hd"]}' for r in order]
    ratios = [r['fold_ratio'] for r in order]

    n_scatter = min(a.n_scatter, len(results))
    scatter_set = {id(order[0]), id(order[-1])} if n_scatter >= 2 else {id(order[0])}
    to_scatter = [r for r in order if id(r) in scatter_set][:n_scatter]

    fig = plt.figure(figsize=(7.2, 3.0 + 2.8 * bool(to_scatter)))
    height_ratios = [1.0, 1.4] if to_scatter else [1.0]
    gs = GridSpec(1 + bool(to_scatter), max(len(to_scatter), 1), figure=fig,
                  height_ratios=height_ratios, hspace=0.55, wspace=0.35)

    axbar = fig.add_subplot(gs[0, :])
    colors = ['#9C4A2F' if r < 1.0 else '#1F3B5C' for r in ratios]
    axbar.bar(range(len(labels)), ratios, color=colors, edgecolor='k', linewidth=0.4, width=0.6)
    axbar.axhline(1.0, ls=':', lw=0.9, color='k')
    axbar.text(len(labels) - 0.5, 1.03, 'fold ratio = 1 (neighbour distance)', fontsize=7,
               ha='right', va='bottom')
    axbar.set_xticks(range(len(labels))); axbar.set_xticklabels(labels, fontsize=7.5)
    axbar.set_ylabel('fold ratio\n(group-image dist. / neighbour dist.)', fontsize=7.5)
    axbar.set_title('Folded codes collapse a position onto its group image: fold ratio < 1 '
                    'only when the code folds', fontsize=8.5)
    for i, r in enumerate(ratios):
        axbar.text(i, r + 0.05, f'{r:.2f}', ha='center', fontsize=7)
    axbar.text(-0.08, 1.05, 'a', transform=axbar.transAxes, fontsize=9, fontweight='bold',
               va='bottom', ha='right')

    letters = 'bcdefgh'
    for j, r in enumerate(to_scatter):
        ax = fig.add_subplot(gs[1, j], projection='3d')
        panel(ax, r['H3'], r['phase'], f'{r["cond"]}/{r["hd"]}: population manifold\n'
              f'(colour = C2 phase, illustrative only)', 'coolwarm')
        ax.text2D(-0.05, 1.0, letters[j], transform=ax.transAxes, fontsize=9,
                  fontweight='bold', va='bottom', ha='right')

    fig.suptitle('The fold, seen geometrically', fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=350)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
