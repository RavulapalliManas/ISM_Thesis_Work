#!/usr/bin/env python3
"""Can the code tell WHICH of two C2-equivalent positions the agent occupies?

This is the test that `odd` power cannot do on its own. Odd isotypic power drops
whenever the HD encoding reshapes the rate-map spectrum -- it drops in s1 too, where
there is no landmark symmetry to fold onto. Phase decoding is immune to that: under
genuine folding onto the quotient X/G the hidden state satisfies

    h(x) == h(R^2 x)

so no classifier, linear or otherwise, can recover which element of the orbit {x, R^2 x}
produced it. Chance is 50%. A global change in spectral shape leaves phase decodable.

Three readouts per model:

  phase_acc   orbit-held-out accuracy of a linear decoder for phase.   Folding -> 0.5
  raw_r2      R^2 of a linear decoder for the raw position (x, y).     Folding -> ~0
  domain_r2   R^2 of a linear decoder for the position within the fundamental domain.

`domain_r2` is NOT a neutral positive control -- do not read it as one. The fundamental
domain coordinate is a *fold* of the raw one, so recovering it from an unfolded code
needs a nonlinear map and a linear decoder scores ~0. It is a second folding indicator,
pointing the opposite way:

    unfolded code:  phase_acc -> 1     raw_r2 high    domain_r2 low
    folded   code:  phase_acc -> 0.5   raw_r2 low     domain_r2 high
    dead     code:  phase_acc -> 0.5   raw_r2 low     domain_r2 low     <- the failure mode

So the genuine sanity check is `max(raw_r2, domain_r2)`: space is still encoded *somehow*.
A model whose phase collapses to chance while BOTH R^2 collapse has simply stopped
representing space, and its chance-level phase means nothing.

Orbits are held out entirely across CV folds, so the decoder must generalise a phase
direction to positions it has never seen -- not memorise per-orbit idiosyncrasies.

The C2 element R^2 (180 degrees) is used for every condition, including s4: it lies in
C4, and it is exactly the group element whose characters `odd` measures.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import (collect, identify,  # noqa: E402
                                                     model_from_checkpoint)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

ARENA = 18
GROUP_ORDER = {'c2': 2, 'c4': 4}          # chance accuracy is 1/order


def rot90(xy: np.ndarray) -> np.ndarray:
    """Quarter turn on 1-based grid coords: (x, y) -> (y, N+1-x). Order 4."""
    return np.stack([xy[:, 1], (ARENA + 1) - xy[:, 0]], 1)


def rot180(xy: np.ndarray) -> np.ndarray:
    """1-based grid coords, so x + x' = ARENA + 1."""
    return (ARENA + 1) - xy


def images(pos: np.ndarray, group: str) -> np.ndarray:
    """(n, |G|, 2) -- the orbit of every position, element 0 being the identity."""
    if group == 'c2':
        return np.stack([pos, rot180(pos)], 1)
    if group == 'c4':
        r1 = rot90(pos)
        r2 = rot90(r1)
        return np.stack([pos, r1, r2, rot90(r2)], 1)
    raise ValueError(f'unknown group {group!r}')


def orbit_and_phase(pos: np.ndarray, group: str = 'c2') -> tuple[np.ndarray, np.ndarray]:
    """Orbit id (its canonical representative) and phase (which group element maps the
    canonical representative onto this position).

    ARENA is even, so no rotation has an integer fixed point and every orbit has exactly
    |G| distinct cells -- no positions need excluding, and chance is exactly 1/|G|.
    """
    im = images(pos, group)                                    # (n, |G|, 2)
    order = im.shape[1]
    for j in range(1, order):
        assert not np.any((im[:, j] == pos).all(1)), f'rotation {j} has a fixed point'
    key = im[:, :, 0] * (ARENA + 1) + im[:, :, 1]              # lexicographic rank
    phase = key.argmin(axis=1)
    orbit = key.min(axis=1)
    return orbit, phase


def canonical(pos: np.ndarray, group: str = 'c2') -> np.ndarray:
    """Position within the fundamental domain of `group`."""
    im = images(pos, group)
    key = im[:, :, 0] * (ARENA + 1) + im[:, :, 1]
    return im[np.arange(len(pos)), key.argmin(axis=1)]


def _balance(y: np.ndarray, rng) -> np.ndarray:
    """Indices with equal counts per phase, so chance is exactly 1/|G|."""
    classes = np.unique(y)
    n = min((y == c).sum() for c in classes)
    keep = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in classes])
    rng.shuffle(keep)
    return keep


def decode(hidden, pos, group='c2', n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    orbit, phase = orbit_and_phase(pos, group)
    keep = _balance(phase, rng)
    H, y, g = hidden[keep], phase[keep], orbit[keep]
    canon = canonical(pos, group)[keep]
    raw = pos[keep]

    accs, dom_r2, raw_r2 = [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(H, y, groups=g):
        sc = StandardScaler().fit(H[tr])
        Xtr, Xte = sc.transform(H[tr]), sc.transform(H[te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        accs.append(clf.score(Xte, y[te]))
        dom_r2.append(Ridge(alpha=1.0).fit(Xtr, canon[tr]).score(Xte, canon[te]))
        raw_r2.append(Ridge(alpha=1.0).fit(Xtr, raw[tr]).score(Xte, raw[te]))
    return float(np.mean(accs)), float(np.mean(raw_r2)), float(np.mean(dom_r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--ckpt-name', default='ckpt_final.pt')
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--group', default='c2', choices=['c2', 'c4'],
                    help='symmetry group to quotient by. c4 is needed to see a C4-folded code.')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob(a.ckpt_name))
    if not ckpts:
        raise SystemExit(f'no {a.ckpt_name} under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    datasets, rows = {}, []
    for i, path in enumerate(ckpts, 1):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(path, ck['meta'])
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, device)
        hidden, pos = collect(model, datasets[cond], hd, a.n_states, device)
        acc, raw_r2, dom_r2 = decode(hidden, pos, group=a.group)
        rows.append({'condition': cond, 'hd_mode': hd, 'k': ck['meta']['k'], 'seed': seed,
                     'group': a.group, 'chance': 1.0 / GROUP_ORDER[a.group],
                     'phase_acc': acc, 'raw_r2': raw_r2, 'domain_r2': dom_r2,
                     'spatial': max(raw_r2, dom_r2), 'path': str(path)})
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/k{ck["meta"]["k"]}/seed{seed:02d}  '
              f'phase={acc:.4f}  raw_r2={raw_r2:+.3f}  domain_r2={dom_r2:+.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
