"""Per-unit spatial rate maps from trained checkpoints, saved as arrays for plotting.

Computes the mean activation of each hidden unit at each arena cell (a place-cell rate map),
ranks units by Skaggs spatial information, and saves the top units per condition to an npz.
Rendering is done separately, in the paper's figure style, so the compute and the look are
kept apart. Nothing here decides how a figure looks; it only produces the maps behind it.

    python3 rate_maps.py --runs /root/runs/hd_invariance \
        --conds s1/full s2/full s2/axis s4/full s4/const --seed 0 --out /root/rate_maps.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import (collect, model_from_checkpoint)  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def mean_rate_maps(hidden, pos, arena=18):
    """(n_states, n_units), 1-indexed (col, row) positions -> mean rate map per unit.

    Returns maps (n_units, arena, arena) indexed [unit, row, col] and occupancy counts
    (arena, arena). Cells never visited have zero count; the caller masks them.
    """
    r, c = pos[:, 1] - 1, pos[:, 0] - 1
    flat = r * arena + c
    n_cells, n_units = arena * arena, hidden.shape[1]
    acc = np.zeros((n_cells, n_units))
    np.add.at(acc, flat, hidden)
    cnt = np.bincount(flat, minlength=n_cells).astype(float)
    rate = acc / np.maximum(cnt[:, None], 1.0)
    maps = rate.T.reshape(n_units, arena, arena)
    return maps, cnt.reshape(arena, arena)


def spatial_information(maps, occ):
    """Skaggs bits per spike, per unit. occ is visit counts; unvisited cells are excluded."""
    p = occ / occ.sum()
    flatp = p.reshape(-1)
    r = maps.reshape(maps.shape[0], -1)
    rbar = (r * flatp[None, :]).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = r / rbar[:, None]
        term = flatp[None, :] * ratio * np.log2(ratio)
    term[~np.isfinite(term)] = 0.0
    term[:, flatp == 0] = 0.0
    return term.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--conds', nargs='+', required=True, help='e.g. s1/full s2/axis')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    datasets, out = {}, {}
    for spec in a.conds:
        cond, hd = spec.split('/')
        ckpt = Path(a.runs) / cond / hd / f'seed_{a.seed:02d}' / 'ckpt_final.pt'
        if not ckpt.exists():
            print(f'  MISSING {ckpt}, skipping'); continue
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        hidden, pos = collect(model, datasets[cond], hd, a.n_states, dev)
        maps, occ = mean_rate_maps(hidden, pos, a.arena)
        si = spatial_information(maps, occ)
        order = np.argsort(si)[::-1][:a.top]
        key = f'{cond}_{hd}'
        out[f'{key}__maps'] = maps[order].astype(np.float32)
        out[f'{key}__si'] = si[order].astype(np.float32)
        out[f'{key}__occ'] = occ.astype(np.float32)
        out[f'{key}__units'] = order.astype(np.int64)
        print(f'  {spec}: {len(pos)} states, top SI = {si[order][:3].round(3)}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out} ({len(out)//4} conditions)')


if __name__ == '__main__':
    main()
