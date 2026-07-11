"""The fold as a remapping failure, in the standard population-vector metric.

Global remapping is measured in the hippocampus as the correlation between the population
representations of two environments (Leutgeb et al. 2005): near zero when the code reorganises
(remaps), near one when it does not. Here the "two environments" are a position and its
symmetry-image within one arena. For each orbit pair (x, R^2 x) we correlate the population
vectors -- the per-position mean hidden state -- across units:

    PV_corr high  ->  x and its image share one representation: the code has NOT separated them
                      (folded onto the quotient). An invariant compass cannot drive remapping.
    PV_corr low   ->  distinct representations: the code has separated the symmetric locations,
                      the network-internal analogue of global remapping.

We report the median PV correlation between orbit-mates per condition, alongside the control
correlation between random non-orbit position pairs. Folding (axis/const in the symmetric arenas)
should push the orbit-mate correlation toward the random baseline's opposite: high where random
pairs are low.

    python3 remapping.py --runs <hd_invariance> --data-root <traj> --out <remapping.csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import collect, identify, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_tda import position_means  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def _corr_rows(A, B):
    """Row-wise Pearson correlation between two (n, U) matrices."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A**2).sum(1) * (B**2).sum(1))
    return num / np.where(den > 1e-9, den, np.nan)


def pv_correlations(means, cells, arena, rng):
    lut = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    pairs = [(i, lut[(arena + 1 - int(x), arena + 1 - int(y))])
             for i, (x, y) in enumerate(cells)
             if (arena + 1 - int(x), arena + 1 - int(y)) in lut]
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    orbit = _corr_rows(means[a], means[b])
    perm = rng.permutation(len(means))                      # random non-orbit baseline
    rand = _corr_rows(means, means[perm])
    return float(np.nanmedian(orbit)), float(np.nanmedian(rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    ckpts = [p for p in ckpts if identify(p, torch.load(p, map_location='cpu',
             weights_only=False)['meta'])[0] in a.conds]
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        if cond not in ds:
            ds[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        H, pos = collect(model, ds[cond], hd, a.n_states, dev)
        means, cells = position_means(H, pos)
        orbit, rand = pv_correlations(means, cells, a.arena, np.random.default_rng(seed))
        rows.append({'condition': cond, 'hd_mode': hd, 'seed': seed,
                     'pv_orbit': orbit, 'pv_random': rand})
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/seed{seed}: pv_orbit={orbit:+.3f} '
              f'pv_random={rand:+.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\n{"cond/hd":<12s} {"pv_orbit":>9s} {"pv_random":>10s}')
    for cond in a.conds:
        for hd in ('full', 'parity', 'axis', 'const'):
            sub = [r for r in rows if r['condition'] == cond and r['hd_mode'] == hd]
            if sub:
                print(f'{cond+"/"+hd:<12s} {np.mean([r["pv_orbit"] for r in sub]):>9.3f} '
                      f'{np.mean([r["pv_random"] for r in sub]):>10.3f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
