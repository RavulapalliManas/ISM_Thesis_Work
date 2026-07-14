"""Export the population manifold behind the geometry figure.

`manifold_s2.png` was an ORPHAN: it is a main-text figure (the fold, seen geometrically) and no
script in the repository produced it. It was also a raster PNG in an otherwise-vector paper. This
script regenerates its contents from the checkpoints and writes them to CSV, so the figure becomes
a function of the data like every other one, and can be emitted as vector.

Two things are exported per network:

  fold ratio    the distance between a cell's population state and that of its 180-degree image,
                in units of the distance to an adjacent cell. Below 1 means the orbit partners sit
                closer together than spatial neighbours do, i.e. the code has folded. Computed in
                the FULL hidden space, so it is not an artefact of any projection.

  manifold      the position-conditioned mean states projected to three principal components, with
                each cell's C2 orbit phase, for plotting.

    PYTHONPATH=. python3 analysis/export_manifold.py --ckpt-root <dir> --data-root <dir> \
        --out-dir Report/data
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import images, ARENA  # noqa: E402
from project5_symmetry.analysis.isometry_quotient import position_means  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

# the four cells the geometry figure contrasts: same encoding with and without a symmetry to fold
# onto, and the two non-invariant encodings in the symmetric arena
PANELS = [('s2', 'axis'), ('s1', 'axis'), ('s2', 'parity'), ('s2', 'full')]


def fold_ratio(M, cells):
    """<d(x, R^2 x)> / <d(x, neighbour)>, in the full hidden space."""
    idx = {tuple(c): i for i, c in enumerate(cells)}
    orb = images(cells, 'c2', ARENA).astype(int)
    part, nb = [], []
    for i, c in enumerate(cells):
        j = idx.get(tuple(orb[i, 1]))
        if j is not None and j != i:
            part.append(np.linalg.norm(M[i] - M[j]))
        for d in ((0, 1), (1, 0)):
            k = idx.get((c[0] + d[0], c[1] + d[1]))
            if k is not None:
                nb.append(np.linalg.norm(M[i] - M[k]))
    return float(np.mean(part) / (np.mean(nb) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--embed-seed', type=int, default=0)
    ap.add_argument('--n-traj', type=int, default=600)
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)
    out = Path(a.out_dir)

    ds = {}
    for c in sorted({c for c, _ in PANELS}):
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    ratios, coords = [], []
    for cond, hd in PANELS:
        for s in a.seeds:
            p = ck / 'hd_invariance' / cond / hd / f'seed_{s:02d}' / 'ckpt_final.pt'
            if not p.exists():
                continue
            model = model_from_checkpoint(
                torch.load(p, map_location='cpu', weights_only=False), dev)
            H, pos = collect(model, ds[cond], hd, a.n_states, dev)
            M, cells = position_means(H, pos)
            r = fold_ratio(M, cells)
            ratios.append({'condition': cond, 'hd_mode': hd, 'seed': s, 'fold_ratio': round(r, 4)})
            print(f'  {cond}/{hd}/s{s:02d}  fold ratio = {r:.3f}', flush=True)

            if s == a.embed_seed:
                X = M - M.mean(0)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                P = X @ Vt[:3].T
                orb = images(cells, 'c2', ARENA).astype(int)
                # orbit phase: 0 for the canonical member of each pair, 1 for its image
                phase = [int(tuple(c) > tuple(orb[i, 1])) for i, c in enumerate(cells)]
                var3 = float(S[:3].sum() ** 2 / (S ** 2).sum()) if S.sum() else 0.0
                for i, c in enumerate(cells):
                    coords.append({
                        'condition': cond, 'hd_mode': hd, 'seed': s,
                        'x': int(c[0]), 'y': int(c[1]), 'phase': phase[i],
                        'pc1': round(float(P[i, 0]), 4), 'pc2': round(float(P[i, 1]), 4),
                        'pc3': round(float(P[i, 2]), 4),
                        'fold_ratio': round(r, 4),
                        'var3': round(float((S[:3] ** 2).sum() / (S ** 2).sum()), 4)})

    for name, rows in (('manifold_fold_ratio.csv', ratios), ('manifold_coords.csv', coords)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {out / name}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
