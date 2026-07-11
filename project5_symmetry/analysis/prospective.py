"""Prospective (anticipatory) coding: do fields peak ahead of the animal, and does it need prediction?

Hippocampal place cells fire prospectively -- their activity at time t is better explained by where
the animal is ABOUT to be than by where it is (a signature of predictive coding). We test this, and
tie it to the objective, by rebuilding each unit's rate map keyed on the position at time t+delta
for a range of temporal offsets delta, and asking which offset makes the population's spatial tuning
sharpest.

    delta* > 0   activity leads position  -> PROSPECTIVE (anticipatory) coding
    delta* = 0   activity tracks the present position
    delta* < 0   activity lags position (retrospective)

Run across prediction horizons k = 0, 1, 3, 5 (k=0 is the pure autoencoder). If prospective coding
is a consequence of predicting the future, delta* should sit at 0 for the autoencoder and move
forward as k grows. Measured on the non-folding `full` encoding so clean place fields are available.

    python3 prospective.py --runs <k0> <k1> <k3> <k5> --data-root <traj> --hd full --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.rate_maps import mean_rate_maps, spatial_information  # noqa: E402
from project5_symmetry.analysis.run_spectrum import identify, model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


@torch.no_grad()
def collect_trajs(model, ds, hd_mode, n_traj, device, seed=0):
    """List of (h_traj (T,U), pos_traj (T,2)) kept contiguous so positions can be time-shifted."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for idx in torch.randperm(len(ds), generator=g)[:n_traj]:
        obs, act, pos, _ = ds[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        p = pos.numpy()[:h.shape[0]].astype(int)
        out.append((h, p))
    return out


def si_at_delta(trajs, delta, arena, n_top=100):
    """Mean spatial information of the top-n_top variable units, rate maps keyed on pos[t+delta]."""
    Hs, Ps = [], []
    for h, p in trajs:
        T = min(len(h), len(p))
        lo, hi = max(0, -delta), min(T, T - delta)
        if hi - lo < 2:
            continue
        Hs.append(h[lo:hi])
        Ps.append(p[lo + delta:hi + delta])
    H, P = np.concatenate(Hs, 0), np.concatenate(Ps, 0)
    maps, occ = mean_rate_maps(H, P, arena)
    si = spatial_information(maps, occ)
    top = np.argsort(H.var(0))[-n_top:]
    return float(np.mean(si[top]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='one or more ckpt roots (any k)')
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--cond', default='s2')
    ap.add_argument('--hd', default='full')
    ap.add_argument('--deltas', nargs='+', type=int, default=[-3, -2, -1, 0, 1, 2, 3])
    ap.add_argument('--n-traj', type=int, default=120)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ds = TrajectoryDataset(str(Path(a.data_root) / a.cond))
    ckpts = []
    for root in a.runs:
        for p in sorted(Path(root).rglob('ckpt_final.pt')):
            ck = torch.load(p, map_location='cpu', weights_only=False)
            cond, hd, seed = identify(p, ck['meta'])
            if cond == a.cond and hd == a.hd:
                ckpts.append((p, ck, seed))
    print(f'{len(ckpts)} {a.cond}/{a.hd} checkpoints', flush=True)

    rows = []
    for i, (p, ck, seed) in enumerate(ckpts, 1):
        k = int(ck['meta']['k'])
        model = model_from_checkpoint(ck, dev)
        trajs = collect_trajs(model, ds, a.hd, a.n_traj, dev, seed=seed)
        si = {d: si_at_delta(trajs, d, a.arena) for d in a.deltas}
        dstar = max(si, key=si.get)
        # info-weighted mean offset, a smooth version of delta*
        w = np.array([max(0.0, si[d]) for d in a.deltas]); w = w / w.sum()
        centroid = float(np.sum(w * np.array(a.deltas)))
        row = {'k': k, 'hd': a.hd, 'seed': seed, 'delta_star': dstar, 'delta_centroid': centroid}
        row.update({f'si_{d:+d}': si[d] for d in a.deltas})
        rows.append(row)
        print(f'  [{i}/{len(ckpts)}] k={k} seed={seed}: delta*={dstar:+d} '
              f'centroid={centroid:+.2f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\n{"k":>3s} {"n":>3s} {"delta*_mean":>12s} {"centroid":>9s}')
    for k in sorted({r['k'] for r in rows}):
        sub = [r for r in rows if r['k'] == k]
        print(f'{k:>3d} {len(sub):>3d} {np.mean([r["delta_star"] for r in sub]):>12.2f} '
              f'{np.mean([r["delta_centroid"] for r in sub]):>9.2f}')
    print(f'\nwrote {a.out}  (delta* > 0 = prospective/anticipatory)')


if __name__ == '__main__':
    main()
