"""Is the topology-before-geometry NULL an artefact of linear (PCA) dimensionality reduction?

The main analysis reads persistent H1 off a PCA-6 reduction (cosine metric, shuffle null) and does
not recover the arenas' loop topology. Because six linear components capture only ~half the
variance here, a loop embedded nonlinearly could be invisible to PCA yet present -- so we recompute
the Betti-1 estimate under distance-preserving nonlinear embeddings (Isomap, Laplacian/spectral)
and under more linear components, on the same position-conditioned means. If the nonlinear
embeddings recover b1 = expected where PCA-6 does not, the null was a linear-readout artefact and
the topology IS there (nonlinearly embedded); if they also fail, the null is robust to the choice
of reduction.

    python3 topology_robustness.py --runs <topology_ckpts> --data-root <topology_traj> --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_tda import position_means  # noqa: E402
from project5_symmetry.analysis.tda import betti1  # noqa: E402
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.environments.topology_arenas import (EXPECTED_B1,  # noqa: E402
                                                            LAYOUTS,
                                                            make_topology_env)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def embeddings(means, seed=0):
    """Return {name: (Y, metric)} for the reductions to compare."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, SpectralEmbedding
    n = len(means)
    nn = min(12, n - 1)
    out = {}
    out['pca6'] = (PCA(n_components=6, whiten=True, random_state=seed).fit_transform(means), 'cosine')
    out['pca20'] = (PCA(n_components=min(20, n - 1), whiten=True, random_state=seed)
                    .fit_transform(means), 'cosine')
    out['isomap6'] = (Isomap(n_components=6, n_neighbors=nn).fit_transform(means), 'euclidean')
    try:
        out['spectral6'] = (SpectralEmbedding(n_components=6, n_neighbors=nn,
                                              random_state=seed).fit_transform(means), 'euclidean')
    except Exception as e:                                  # spectral can fail on odd graphs
        print(f'    spectral embedding failed: {e}', flush=True)
    return out


def ensure_topology_data(layout, data_root, n_traj, workers):
    d = Path(data_root) / layout
    generate_dataset(make_topology_env(layout, F=7, seed=0), n_traj=n_traj, T=200,
                     out_dir=str(d), n_workers=workers, desc=layout,
                     env_factory=make_topology_env,
                     factory_kwargs={'layout': layout, 'F': 7, 'seed': 0})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--n-traj', type=int, default=300)
    ap.add_argument('--n-states', type=int, default=15_000)
    ap.add_argument('--n-shuffles', type=int, default=16)
    ap.add_argument('--dataset-workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        layout, seed = ck['meta']['layout'], ck['meta']['seed']
        if layout not in LAYOUTS:
            continue
        if layout not in ds:
            ds[layout] = TrajectoryDataset(str(ensure_topology_data(
                layout, a.data_root, a.n_traj, a.dataset_workers)))
        model = model_from_checkpoint(ck, dev)
        H, pos = collect(model, ds[layout], 'full', a.n_states, dev)
        means, cells = position_means(H, pos)
        if not np.all(np.isfinite(means)) or len(means) < 20:
            continue
        row = {'layout': layout, 'seed': seed, 'b1_true': EXPECTED_B1[layout], 'n_cells': len(means)}
        for name, (Y, metric) in embeddings(means, seed).items():
            b1, conf = betti1(Y, metric=metric, n_shuffles=a.n_shuffles, seed=0)
            row[f'b1_{name}'] = b1
        rows.append(row)
        got = '  '.join(f'{k[3:]}={row[k]}' for k in row if k.startswith('b1_') and k != 'b1_true')
        print(f'  [{i}/{len(ckpts)}] {layout:<9s} seed{seed} b1_true={row["b1_true"]}  {got}',
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    methods = [k for k in rows[0] if k.startswith('b1_') and k != 'b1_true']
    print(f'\nfraction of seeds recovering b1_true, by layout x method:')
    print('layout      b1_true  ' + '  '.join(m[3:] for m in methods))
    for lay in LAYOUTS:
        sub = [r for r in rows if r['layout'] == lay]
        if not sub:
            continue
        fr = [np.mean([int(r[m] == r['b1_true']) for r in sub]) for m in methods]
        print(f'{lay:<11s} {sub[0]["b1_true"]:>7d}  ' + '  '.join(f'{x:>5.2f}' for x in fr))
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
