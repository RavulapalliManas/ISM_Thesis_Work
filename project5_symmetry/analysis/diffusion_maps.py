"""Diffusion-map geometry of the learned manifolds.

Diffusion maps (Coifman & Lafon 2006) embed a point cloud through the eigenfunctions of a Markov
random walk built on it, so distances in the embedding are diffusion distances -- a
geometry-aware, density-robust metric -- and the eigenvalue spectrum gives a principled
intrinsic-dimension readout (a spectral gap after the first few coordinates = a low-dimensional
manifold). We use it two ways on the position-conditioned population means:

  topology arenas : does the diffusion embedding trace the arena's loop? We score it by how well
                    the arena angle is parametrised by the first two non-trivial diffusion
                    coordinates (circular correlation), and report the spectral gap.
  symmetry fold   : the fold as geometry -- the diffusion distance between a position and its
                    group image, in units of the distance to a spatial neighbour. < 1 = folded.

Writes a metrics CSV and a figure with both panels.

    python3 diffusion_maps.py --topo-runs <topology> --topo-data <topology_traj> \
        --fold-runs <hd_invariance> --fold-data <symmetry_traj> --out-csv <csv> --out-fig <pdf>
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
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def diffusion_map(X, n_coords=8, alpha=1.0, eps_scale=1.0, t=1):
    """Return (coords (n, n_coords), eigenvalues (n_coords,)). alpha=1 -> Laplace-Beltrami."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    eps = eps_scale * np.median(D[D > 0]) ** 2
    K = np.exp(-(D ** 2) / eps)
    q = K.sum(1)
    K = K / np.outer(q ** alpha, q ** alpha)             # density normalisation
    d = K.sum(1)
    P = K / d[:, None]                                   # row-stochastic Markov matrix
    # symmetric conjugate has the same spectrum and real eigenvectors
    s = np.sqrt(d)
    Psym = P * (s[:, None] / s[None, :])
    Psym = (Psym + Psym.T) / 2
    w, v = np.linalg.eigh(Psym)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    psi = v / s[:, None]                                  # right eigenvectors of P
    k = min(n_coords + 1, psi.shape[1])
    coords = psi[:, 1:k] * (w[1:k] ** t)                  # drop trivial 1st component
    return coords, w[1:k]


def _circular_corr(theta, phi):
    """Max over rotation/reflection of |corr(cos(theta - phi + a))| -- how well phi tracks theta."""
    best = 0.0
    for sign in (1, -1):
        c = np.corrcoef(np.cos(theta), np.cos(sign * phi))[0, 1]
        s = np.corrcoef(np.cos(theta), np.sin(sign * phi))[0, 1]
        best = max(best, np.hypot(c, s))
    return float(best)


def _orbit_pairs(cells, arena):
    lut = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    img, nbr = [], []
    for i, (x, y) in enumerate(cells):
        img.append(lut.get((arena + 1 - int(x), arena + 1 - int(y)), -1))
        dd = np.abs(cells - cells[i]).sum(1); dd[i] = 10**9
        nbr.append(int(dd.argmin()))
    return np.array(img), np.array(nbr)


def _load_means(ckpt, ds, hd, n_states, dev):
    model = model_from_checkpoint(ckpt, dev)
    H, pos = collect(model, ds, hd, n_states, dev)
    return position_means(H, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topo-runs', required=True)
    ap.add_argument('--topo-data', required=True)
    ap.add_argument('--fold-runs', required=True)
    ap.add_argument('--fold-data', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--n-states', type=int, default=15_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-fig', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    topo = ['open', 'annulus', 'theta', 'figure8']
    fold = [('s2', 'full'), ('s2', 'parity'), ('s2', 'axis')]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))

    for j, lay in enumerate(topo):
        p = Path(a.topo_runs) / lay / f'seed_{a.seed:02d}' / 'ckpt_final.pt'
        ck = torch.load(p, map_location='cpu', weights_only=False)
        ds = TrajectoryDataset(str(Path(a.topo_data) / lay))
        means, cells = _load_means(ck, ds, 'full', a.n_states, dev)
        coords, evals = diffusion_map(means)
        ctr = cells.astype(float).mean(0)
        ang = np.arctan2(cells[:, 1] - ctr[1], cells[:, 0] - ctr[0])
        phi = np.arctan2(coords[:, 1], coords[:, 0])
        loop = _circular_corr(ang, phi)
        gap = float(evals[0] / evals[1]) if len(evals) > 1 and evals[1] > 0 else float('nan')
        idim = int(np.sum(evals > 0.5 * evals[0]))
        rows.append({'kind': 'topology', 'name': lay, 'hd': 'full', 'loop_score': loop,
                     'intrinsic_dim': idim, 'spectral_gap_1_2': gap,
                     **{f'eval_{i}': float(evals[i]) for i in range(min(6, len(evals)))}})
        ax = axes[0, j]
        ax.scatter(coords[:, 0], coords[:, 1], c=ang, cmap='twilight', s=8, edgecolors='none')
        ax.set_title(f'{lay}  loop={loop:.2f}', fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    for j, (cond, hd) in enumerate(fold):
        p = Path(a.fold_runs) / cond / hd / f'seed_{a.seed:02d}' / 'ckpt_final.pt'
        ck = torch.load(p, map_location='cpu', weights_only=False)
        ds = TrajectoryDataset(str(Path(a.fold_data) / cond))
        means, cells = _load_means(ck, ds, hd, a.n_states, dev)
        coords, evals = diffusion_map(means)
        img, nbr = _orbit_pairs(cells, a.arena)
        ok = img >= 0
        di = np.linalg.norm(coords[ok] - coords[img[ok]], axis=1)
        dn = np.linalg.norm(coords[ok] - coords[nbr[ok]], axis=1)
        ratio = float(np.nanmedian(di / np.where(dn > 1e-12, dn, np.nan)))
        phase = ((cells[:, 0] + cells[:, 1]) % 2)          # a simple 2-colour orbit tag for display
        rows.append({'kind': 'fold', 'name': cond, 'hd': hd, 'fold_ratio_diffusion': ratio,
                     'intrinsic_dim': int(np.sum(evals > 0.5 * evals[0])),
                     **{f'eval_{i}': float(evals[i]) for i in range(min(6, len(evals)))}})
        ax = axes[1, j]
        ax.scatter(coords[:, 0], coords[:, 1], c=phase, cmap='coolwarm', s=8, edgecolors='none')
        ax.set_title(f'{cond}/{hd}  fold={ratio:.2f}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 3].axis('off')
    fig.suptitle('Diffusion-map embeddings: topology arenas (top), C2 fold (bottom)', fontsize=9)
    fig.tight_layout()
    Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_fig, dpi=200); plt.close(fig)

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('topology loop scores + fold diffusion ratios:')
    for r in rows:
        if r['kind'] == 'topology':
            print(f'  {r["name"]:<9s} loop={r["loop_score"]:.2f} idim={r["intrinsic_dim"]}')
        else:
            print(f'  {r["name"]}/{r["hd"]:<7s} fold_ratio_diff={r["fold_ratio_diffusion"]:.2f}')
    print(f'\nwrote {a.out_csv} and {a.out_fig}')


if __name__ == '__main__':
    main()
