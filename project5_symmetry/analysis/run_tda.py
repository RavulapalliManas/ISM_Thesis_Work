#!/usr/bin/env python3
"""Topology and geometry of the population code, at every training checkpoint.

For each checkpoint we build the position-conditioned population manifold -- one point
per passable arena cell, h(x) = E[h | agent at x] -- and measure two things about it:

  TOPOLOGY   b1_hat : Betti number from persistent H1 (ripser, cosine, Z_47, PCA-6), read
                      off the largest gap among bars that outlive a per-cloud SHUFFLE NULL.
                      The null is not optional: a scale-free gap rule cannot return 0, so
                      without it the `open` arena (true b1 = 0) is scored wrong at every
                      checkpoint and the whole sweep is uninterpretable.
             conf   : the last accepted bar as a multiple of that noise floor.
             correct: b1_hat == the arena's true b1.

  GEOMETRY   metric : Spearman correlation between neural distance ||h(x) - h(y)|| and
                      GEODESIC distance in the arena (BFS through passable cells, not
                      Euclidean -- a wall means the two cells are far apart even if they
                      are adjacent on the page).
             decode : cross-validated R^2 of a linear decoder for position.

The claim under test: `correct` saturates at an earlier training step than `metric`.

Geodesic, not Euclidean, is the whole point. In an annulus two cells on opposite sides of
the central block are Euclidean-near and geodesic-far. A code that has learned the arena's
*shape* separates them; a code that has only learned "nearby pixels look alike" does not.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from functools import lru_cache
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.tda import betti1, prepare  # noqa: E402
from project5_symmetry.environments.topology_arenas import (ARENA, EXPECTED_B1,  # noqa: E402
                                                            is_passable)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


@lru_cache(maxsize=None)
def geodesic_matrix(layout):
    """All-pairs shortest path through passable cells (4-connectivity). BFS per source."""
    cells = [(r, c) for r in range(1, ARENA + 1) for c in range(1, ARENA + 1)
             if is_passable(layout, r, c)]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    D = np.full((n, n), np.inf)
    for s, src in enumerate(cells):
        D[s, s] = 0
        q = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (r + dr, c + dc)
                j = idx.get(nb)
                if j is not None and not np.isfinite(D[s, j]):
                    D[s, j] = D[s, idx[(r, c)]] + 1
                    q.append(nb)
    D.flags.writeable = False          # cached: never mutate in place
    return np.array(cells), D


def position_means(hidden, pos, min_count=8):
    keys, inv, counts = np.unique(pos, axis=0, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    means = np.stack([hidden[inv == i].mean(0) for i in range(len(keys))])
    return means[keep], keys[keep]


def metric_fidelity(means, cells, layout):
    """Spearman(neural distance, geodesic distance) over all cell pairs present."""
    grid_cells, D = geodesic_matrix(layout)
    idx = {tuple(c): i for i, c in enumerate(grid_cells)}
    sel = [idx[tuple(c)] for c in cells if tuple(c) in idx]
    keep = [i for i, c in enumerate(cells) if tuple(c) in idx]
    Dg = D[np.ix_(sel, sel)]
    M = means[keep]
    Dn = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
    iu = np.triu_indices(len(sel), k=1)
    g, nrl = Dg[iu], Dn[iu]
    ok = np.isfinite(g)
    return float(spearmanr(g[ok], nrl[ok]).statistic)


def decode_r2(means, cells, n_splits=5):
    X, y = StandardScaler().fit_transform(means), cells.astype(float)
    scores = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=0).split(X):
        scores.append(Ridge(alpha=1.0).fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/topology')
    ap.add_argument('--n-states', type=int, default=30_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--n-shuffles', type=int, default=32,
                    help='shuffle null for the H1 noise floor; 0 restores the broken '
                         'scale-free rule that can never report b1 = 0')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_*.pt'))
    if not ckpts:
        raise SystemExit(f'no checkpoints under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    datasets, rows = {}, []
    for i, path in enumerate(ckpts, 1):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        meta = ck['meta']
        layout, seed, step = meta['layout'], meta['seed'], ck['step']
        variant = meta.get('variant', '')          # init-study runs tag their arm
        b1_true = EXPECTED_B1[layout]
        if layout not in datasets:
            datasets[layout] = TrajectoryDataset(str(Path(a.data_root) / layout))
        # A diverged run (e.g. an unstable initialisation) has non-finite weights. Report it
        # as diverged rather than letting PCA raise on NaN halfway through the sweep.
        if not all(torch.isfinite(v).all() for v in ck['model'].values()):
            rows.append({'layout': layout, 'variant': variant, 'seed': seed, 'step': step,
                         'b1_true': b1_true, 'b1_hat': -1, 'b1_conf': float('nan'),
                         'b1_correct': 0, 'metric': float('nan'), 'decode_r2': float('nan'),
                         'n_cells': 0})
            print(f'  [{i}/{len(ckpts)}] {layout}/{variant} seed{seed} step {step}: '
                  f'NON-FINITE WEIGHTS -- diverged', flush=True)
            continue

        model = model_from_checkpoint(ck, dev)
        hidden, pos = collect(model, datasets[layout], 'full', a.n_states, dev)
        means, cells = position_means(hidden, pos)
        if not np.all(np.isfinite(means)):
            rows.append({'layout': layout, 'variant': variant, 'seed': seed, 'step': step,
                         'b1_true': b1_true, 'b1_hat': -1, 'b1_conf': float('nan'),
                         'b1_correct': 0, 'metric': float('nan'), 'decode_r2': float('nan'),
                         'n_cells': len(cells)})
            print(f'  [{i}/{len(ckpts)}] {layout}/{variant} seed{seed} step {step}: '
                  f'non-finite hidden states -- diverged', flush=True)
            continue

        Y = prepare(means, n_pcs=6, n_points=min(1200, len(means)))
        b1_hat, conf = betti1(Y, metric='cosine', n_shuffles=a.n_shuffles, seed=0)
        rows.append({'layout': layout, 'variant': variant, 'seed': seed,
                     'step': step, 'b1_true': b1_true,
                     'b1_hat': b1_hat, 'b1_conf': conf, 'b1_correct': int(b1_hat == b1_true),
                     'metric': metric_fidelity(means, cells, layout),
                     'decode_r2': decode_r2(means, cells), 'n_cells': len(cells)})
        r = rows[-1]
        print(f'  [{i}/{len(ckpts)}] {layout:<11s} seed{seed} step {str(step):>6s}  '
              f'b1 {b1_hat}/{b1_true} (conf {conf:5.2f})  metric {r["metric"]:+.3f}  '
              f'decode {r["decode_r2"]:+.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
