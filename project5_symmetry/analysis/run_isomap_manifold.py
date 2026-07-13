"""Tier-2 item 7: the manifold IS the quotient -- show it, don't just decode it.

Levenstein et al.'s Fig. 3a-d show an Isomap embedding of the wake population state that
visibly looks like the room the agent explored. The equivalent claim here is geometric, not
just a classifier score: under folding, room B's population states should land ON TOP OF room
A's in the embedding (the network's internal manifold is glued into one copy), and under
lifting they should form two separate lobes.

Two-room compartment maze, matched pair at the same k (translation folds per the Theorem,
rotation does not -- Methods): same architecture, same k, same training budget, only the group
action on heading differs. Room-interior states only (the corridor is not part of the group
action the Theorem covers -- Methods), one representative seed per condition (Isomap embeds a
SINGLE network's manifold; pooling seeds would mix unaligned hidden bases and the picture would
be meaningless).

    python3 run_isomap_manifold.py --runs <horizon_k*/compartment> --data-root <compartment data>
                                    --k 3 --seed 0 --out <figures/fig_isomap_manifold.pdf>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_compartments import collect, label_rooms  # noqa: E402
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.compartment_arenas import SIZE  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

ISOMAP_N_NEIGHBORS = 150     # Levenstein et al.'s parameters
ISOMAP_N_COMPONENTS = 2
ISOMAP_METRIC = 'cosine'
N_STATES = 6000               # room-interior states per condition; enough density for k=150 nn


def embed(runs_dir, data_root, mode, seed, n_states=N_STATES, n_components=ISOMAP_N_COMPONENTS):
    ckpt = Path(runs_dir) / mode / f'seed_{seed:02d}' / 'ckpt_final.pt'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = model_from_checkpoint(ck, torch.device('cpu'))
    ds = TrajectoryDataset(str(Path(data_root) / mode))

    H, P = collect(model, ds, n_states * 3, seed=seed)   # oversample; most states are corridor
    room, loc = label_rooms(P, mode)
    keep = room >= 0
    H, room, loc = H[keep][:n_states], room[keep][:n_states], loc[keep][:n_states]

    from sklearn.manifold import Isomap
    iso = Isomap(n_neighbors=ISOMAP_N_NEIGHBORS, n_components=n_components, metric=ISOMAP_METRIC)
    emb = iso.fit_transform(H)
    return emb, room, loc, ck['meta']


def _bivariate_color(loc):
    """(row, col) within a room -> RGB. Red=row axis, green=col axis, so the SAME local cell
    gets the SAME colour whichever room it came from -- if translation glues the rooms, room
    A's and room B's colours should occupy the SAME region of the embedding."""
    rc = loc.astype(float) / max(SIZE - 1, 1)
    return np.stack([rc[:, 0], rc[:, 1], np.full(len(rc), 0.55)], axis=1)


def matched_cell_distance(emb, room, loc, min_count=3):
    """The diagnostic statistic: for every local cell present in both rooms, the distance
    between room A's and room B's per-cell centroid in the embedding, normalised by the
    embedding's overall spread (so it is comparable across networks/conditions). Small ->
    room B's copy of a cell lands where room A's does (glued, folded). Large -> the two rooms
    occupy separate regions (lifted). This is the direct embedding-space analogue of
    `repetition_index` (which does the same matching in raw hidden-state correlation, not in
    the 2D embedding) -- kept separate because a low-dimensional Isomap embedding can in
    principle look folded/unfolded differently than the full hidden space if 2 components
    lose relevant structure; checking both is the honest thing to do.
    """
    scale = emb.std()
    if scale < 1e-9:
        return float('nan'), 0
    A, la = emb[room == 0], loc[room == 0]
    B, lb = emb[room == 1], loc[room == 1]
    dists = []
    for i in range(SIZE):
        for j in range(SIZE):
            ma = (la[:, 0] == i) & (la[:, 1] == j)
            mb = (lb[:, 0] == i) & (lb[:, 1] == j)
            if ma.sum() >= min_count and mb.sum() >= min_count:
                dists.append(np.linalg.norm(A[ma].mean(0) - B[mb].mean(0)))
    if not dists:
        return float('nan'), 0
    return float(np.mean(dists) / scale), len(dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='.../horizon_k{K}/compartment')
    ap.add_argument('--data-root', required=True, help='.../compartment (translation, rotation)')
    ap.add_argument('--fig-seed', type=int, default=0, help='which seed the FIGURE panels use')
    ap.add_argument('--n-seeds', type=int, default=8, help='seeds 0..n_seeds-1 for the CSV/stats')
    ap.add_argument('--out-fig', required=True)
    ap.add_argument('--out-csv', required=True)
    a = ap.parse_args()

    import csv as csv_mod
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _PREFERRED = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'TeX Gyre Heros',
                 'DejaVu Sans']
    _have = {f.name for f in font_manager.fontManager.ttflist}
    _family = next((f for f in _PREFERRED if f in _have and f != 'Helvetica'), 'DejaVu Sans')
    plt.rcParams.update({
        'font.size': 7, 'font.family': _family, 'mathtext.fontset': 'dejavusans',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'axes.edgecolor': '#222222', 'axes.labelcolor': '#222222', 'text.color': '#222222',
        'xtick.color': '#222222', 'ytick.color': '#222222',
        'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    })
    ROOM_COLOR = {'A': '#1F3B5C', 'B': '#9C4A2F'}

    # -- CSV / stats: every seed, both conditions --
    rows = []
    cache = {}
    for mode in ('translation', 'rotation'):
        for seed in range(a.n_seeds):
            print(f'embedding {mode} seed {seed}...', flush=True)
            emb, room, loc, meta = embed(a.runs, a.data_root, mode, seed)
            cache[(mode, seed)] = (emb, room, loc, meta)
            d, ncells = matched_cell_distance(emb, room, loc)
            rows.append({'mode': mode, 'seed': seed, 'k': meta['k'],
                        'matched_cell_dist_norm': d, 'n_cells_matched': ncells})
            print(f'  {mode}/seed{seed:02d}  matched_cell_dist={d:.4f} sd  ({ncells} cells)',
                  flush=True)

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {a.out_csv}')

    from scipy.stats import mannwhitneyu
    trans = [r['matched_cell_dist_norm'] for r in rows if r['mode'] == 'translation']
    rot = [r['matched_cell_dist_norm'] for r in rows if r['mode'] == 'rotation']
    u, p = mannwhitneyu(trans, rot, alternative='two-sided')
    print(f'\ntranslation matched_cell_dist: {np.mean(trans):.4f} +- {np.std(trans,ddof=1):.4f}  '
          f'(n={len(trans)})')
    print(f'rotation     matched_cell_dist: {np.mean(rot):.4f} +- {np.std(rot,ddof=1):.4f}  '
          f'(n={len(rot)})')
    print(f'Mann-Whitney U two-sided: p={p:.5g}')

    # -- Figure: one representative seed, coloured by room identity for visual clarity --
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.7))
    for ax, mode, letter in zip(axes, ('translation', 'rotation'), 'ab'):
        emb, room, loc, meta = cache[(mode, a.fig_seed)]
        d = next(r['matched_cell_dist_norm'] for r in rows
                if r['mode'] == mode and r['seed'] == a.fig_seed)
        for label, code, marker in (('room A', 0, 'o'), ('room B', 1, '^')):
            sel = room == code
            ax.scatter(emb[sel, 0], emb[sel, 1], c=ROOM_COLOR['A' if code == 0 else 'B'],
                      s=4, marker=marker, alpha=0.5, linewidths=0, label=label)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=8, fontweight='bold',
               va='top', ha='left', color='#222222', family=_family)
        ax.set_xlabel('Isomap dim. 1'); ax.set_ylabel('Isomap dim. 2')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{mode}, k={meta["k"]}  (matched-cell dist {d:.2f} sd)', fontsize=6.5)
    axes[0].legend(loc='lower right', markerscale=3, fontsize=6)

    Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_fig)
    print(f'wrote {a.out_fig}')


if __name__ == '__main__':
    main()
