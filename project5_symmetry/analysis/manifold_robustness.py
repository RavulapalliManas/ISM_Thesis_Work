"""Is the fold an artefact of PCA? Fold ratio across embedding methods.

The fold ratio is the distance between a position's population state and that of its
180-degree image, in units of the distance to an adjacent position:

    fold_ratio = median_x  d(h(x), h(R^2 x)) / d(h(x), h(neighbour(x)))

< 1 means a position and its group image sit closer than neighbours -> the code has folded
onto the C2 quotient; > 1 means they are held apart. It is defined in the FULL hidden space;
the manifold figure only visualises it. Here we recompute it in the full space and in three
reductions -- linear PCA, distance-preserving nonlinear Isomap, and t-SNE -- to show the
conclusion does not depend on the embedding. PCA is used for the figure because the claim is
metric (superposition) and PCA is the only one of these that preserves distances honestly;
this script demonstrates the qualitative result survives the nonlinear methods regardless.

    python3 manifold_robustness.py --runs <hd_invariance_ckpts> --data-root <traj_root> \
        --out <manifold_robustness.csv>
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

# (condition, hd_mode): the fold cases in s2, plus s1/axis as the no-symmetry control.
CASES = [('s2', 'full'), ('s2', 'parity'), ('s2', 'axis'), ('s1', 'axis')]


def _pairs(cells, arena):
    """For each cell index, the index of its 180-degree image and of a spatial neighbour."""
    lut = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    img, nbr = [], []
    for i, (x, y) in enumerate(cells):
        gi = lut.get((arena + 1 - int(x), arena + 1 - int(y)))
        # nearest passable neighbour by grid distance (exclude self)
        d = np.abs(cells - cells[i]).sum(1)
        d[i] = 10**9
        ni = int(d.argmin())
        img.append(gi if gi is not None else -1)
        nbr.append(ni)
    return np.array(img), np.array(nbr)


def fold_ratio(rep, img, nbr):
    valid = img >= 0
    di = np.linalg.norm(rep[valid] - rep[img[valid]], axis=1)
    dn = np.linalg.norm(rep[valid] - rep[nbr[valid]], axis=1)
    r = di / np.where(dn > 1e-9, dn, np.nan)
    return float(np.nanmedian(r))


def embeddings(means):
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, TSNE
    out = {'full': means}
    out['pca3'] = PCA(n_components=3, random_state=0).fit_transform(means)
    n = len(means)
    out['isomap3'] = Isomap(n_components=3, n_neighbors=min(10, n - 1)).fit_transform(means)
    out['tsne2'] = TSNE(n_components=2, init='pca', perplexity=min(30, (n - 1) / 3),
                        random_state=0).fit_transform(means)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    datasets, rows = {}, []
    for cond, hd in CASES:
        ckpt = Path(a.runs) / cond / hd / f'seed_{a.seed:02d}' / 'ckpt_final.pt'
        if not ckpt.exists():
            print(f'  MISSING {ckpt}'); continue
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        H, pos = collect(model, datasets[cond], hd, a.n_states, dev)
        means, cells = position_means(H, pos)
        img, nbr = _pairs(cells, a.arena)
        embs = embeddings(means)
        row = {'condition': cond, 'hd_mode': hd, 'n_cells': len(cells)}
        for name, rep in embs.items():
            row[f'fold_ratio_{name}'] = fold_ratio(np.asarray(rep, float), img, nbr)
        rows.append(row)
        print(f'  {cond}/{hd}: ' + '  '.join(
            f'{k.replace("fold_ratio_","")}={row[k]:.2f}' for k in row if k.startswith('fold_ratio_')),
            flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nwrote {a.out}\n(fold_ratio < 1 = folded; the axis/s2 case should read < 1 in every '
          'column, full/parity/s2 and axis/s1 should read > 1)')


if __name__ == '__main__':
    main()
