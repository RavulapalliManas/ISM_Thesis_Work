"""Population-manifold point clouds for the topology arenas, saved for plotting.

For each arena we build the position-conditioned manifold (one point per passable cell,
the mean hidden state there), reduce it to three PCA dimensions, and save the coordinates
with each point's arena position and its angle around the arena centroid. The angle is the
natural colour for a loop: if the manifold is a ring, angle runs smoothly around it.

    python3 manifold_pc.py --runs /root/runs/multi/topology --data-root /root/data/topology \
        --layouts open annulus theta figure8 --out /root/results8/manifold_pc.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_tda import position_means  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--layouts', nargs='+', default=['open', 'annulus', 'theta', 'figure8'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from sklearn.decomposition import PCA
    dev = torch.device('cpu')
    out = {}
    for layout in a.layouts:
        ckpt = Path(a.runs) / layout / f'seed_{a.seed:02d}' / 'ckpt_final.pt'
        if not ckpt.exists():
            print(f'  MISSING {ckpt}'); continue
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        ds = TrajectoryDataset(str(Path(a.data_root) / layout))
        model = model_from_checkpoint(ck, dev)
        H, pos = collect(model, ds, 'full', a.n_states, dev)
        means, cells = position_means(H, pos)              # (n_cells, hidden), (n_cells, 2)
        p = PCA(n_components=3, whiten=True, random_state=0).fit(means)
        Y = p.transform(means)
        c = cells.astype(float)
        ctr = c.mean(0)
        ang = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
        out[f'{layout}__Y'] = Y.astype(np.float32)
        out[f'{layout}__cells'] = cells.astype(np.int16)
        out[f'{layout}__angle'] = ang.astype(np.float32)
        out[f'{layout}__evr'] = p.explained_variance_ratio_.astype(np.float32)
        print(f'  {layout}: {len(cells)} cells, PCA-3 EVR '
              f'{p.explained_variance_ratio_[:3].round(3)}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
