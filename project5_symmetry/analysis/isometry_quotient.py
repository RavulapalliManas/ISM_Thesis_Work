"""Is the neural manifold isometric to the arena X, or to the quotient X/G?

The paper's central measurement is a negative: an orbit-phase decoder sits at chance. This asks
the positive question instead. If the quotient law is right, a folded code is not a degraded map
of the arena -- it is a faithful map of a different space. So:

    neural geodesic distance  should match  d_X       for an unfolded code, and
    neural geodesic distance  should match  d_{X/G}   for a folded one,

where d_{X/G}(x, y) = min_{g in G} ||x - g.y||, the metric on the quotient.

Neural distances are geodesics on a kNN graph (Isomap), because a 2-D sheet embedded in 500-D is
curved and chordal distance underestimates far pairs. Physical distances are Euclidean. Fit is
Kruskal stress-1 with an optimal scale (isometry is only ever defined up to a global scale, since
neural units are arbitrary):

    a* = argmin_a ||D_n - a D_t||,   stress = ||D_n - a* D_t|| / ||D_n||

PREDICTIONS, stated before the run (see also --predict):

                        vs d_X      vs d_{X/C2}
    s1/axis  (nothing to fold)   LOW        HIGH      <- the reachable null
    s2/full  (unfolded)          LOW        HIGH
    s2/axis  (folded)            HIGH       LOW       <- the fold
    s4/const (folded by C4)      HIGH       LOW (C4)

The s1/axis row is what makes this falsifiable: applying the quotient metric to a network with no
symmetry to exploit must NOT improve its score. If it does, the measure is broken.

Nulls in every row: a position-shuffled ceiling (stress the metric returns when the code carries
no spatial information at all), so every number is read against a reachable worst case.

    PYTHONPATH=. python3 analysis/isometry_quotient.py \
        --ckpt-root <dir> --data-root <dir> --out Report/data/isometry_quotient.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import Isomap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import images, ARENA  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def position_means(H: np.ndarray, pos: np.ndarray):
    """Mean hidden state per arena cell. Returns (M, cells) with cells in a fixed order."""
    key = [tuple(int(v) for v in p) for p in pos]
    acc, cnt = {}, {}
    for k, h in zip(key, H):
        acc[k] = acc.get(k, 0.0) + h
        cnt[k] = cnt.get(k, 0) + 1
    cells = sorted(acc)
    M = np.stack([acc[c] / cnt[c] for c in cells])
    return M, np.array(cells, dtype=int)


def neural_geodesic(M: np.ndarray, k: int) -> np.ndarray:
    """Geodesic distances on the kNN graph over position-conditioned means."""
    iso = Isomap(n_neighbors=k, n_components=2, metric='euclidean')
    iso.fit(M)
    return np.asarray(iso.dist_matrix_)


def sham_images(cells: np.ndarray) -> np.ndarray:
    """A SHAM order-2 group: translate by half the arena, mod ARENA. Same |G| = 2 and the same
    'min over images' compression as C2, but it is NOT a symmetry of any arena here, so no network
    can fold by it.

    This is the control the naive quotient metric needs. d_{X/G} <= d_X pointwise, so ANY group
    shrinks long distances; a code whose neural geodesics saturate at long range therefore scores
    better against d_{X/G} whether or not it folds. Comparing the true group against a sham group
    of the same order removes that artifact: only a real fold should prefer the real symmetry.
    """
    h = ARENA // 2
    shifted = ((cells - 1 + np.array([h, 0])) % ARENA) + 1
    return np.stack([cells, shifted], 1)


def quotient_metric(cells: np.ndarray, group: str | None) -> np.ndarray:
    """d_{X/G}(x, y) = min_{g in G} ||x - g.y||. group=None gives plain d_X."""
    if group is None:
        d = cells[:, None, :] - cells[None, :, :]
        return np.linalg.norm(d, axis=-1)
    orb = (sham_images(cells) if group == 'sham'
           else images(cells, group, ARENA)).astype(float)   # (n, |G|, 2)
    # For each pair (i, j): min over the orbit of j of ||x_i - g.x_j||.
    diff = cells[:, None, None, :] - orb[None, :, :, :]      # (n, n, |G|, 2)
    return np.linalg.norm(diff, axis=-1).min(axis=2)


def fold_coincidence(M: np.ndarray, cells: np.ndarray, group: str) -> float:
    """Median cosine between a cell's mean state and that of its group image. A direct test of the
    fold, independent of any distance metric: ~1 means H(x) == H(g.x), i.e. the orbit is one point.
    """
    idx = {tuple(c): i for i, c in enumerate(cells)}
    orb = images(cells, group, ARENA).astype(int)
    cos = []
    for i, c in enumerate(cells):
        j = idx.get(tuple(orb[i, 1]))
        if j is None or j == i:
            continue
        u, v = M[i], M[j]
        cos.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    return float(np.median(cos)) if cos else float('nan')


def stress(Dn: np.ndarray, Dt: np.ndarray) -> float:
    """Kruskal stress-1 of the target metric against the neural one, with an optimal scale."""
    iu = np.triu_indices_from(Dn, k=1)
    dn, dt = Dn[iu], Dt[iu]
    ok = np.isfinite(dn) & np.isfinite(dt)
    dn, dt = dn[ok], dt[ok]
    a = float(dn @ dt / (dt @ dt)) if dt @ dt > 0 else 0.0
    return float(np.linalg.norm(dn - a * dt) / (np.linalg.norm(dn) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--hds', nargs='+', default=['full', 'parity', 'axis', 'const'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--n-traj', type=int, default=600)
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)
    rng = np.random.default_rng(0)

    ds = {}
    for c in a.conds:
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    for cond in a.conds:
        for hd in a.hds:
            for s in a.seeds:
                p = ck / 'hd_invariance' / cond / hd / f'seed_{s:02d}' / 'ckpt_final.pt'
                if not p.exists():
                    continue
                model = model_from_checkpoint(
                    torch.load(p, map_location='cpu', weights_only=False), dev)
                H, pos = collect(model, ds[cond], hd, a.n_states, dev)
                M, cells = position_means(H, pos)
                Dn = neural_geodesic(M, a.knn)

                d_x = quotient_metric(cells, None)
                d_c2 = quotient_metric(cells, 'c2')
                d_c4 = quotient_metric(cells, 'c4')
                d_sham = quotient_metric(cells, 'sham')

                # Null: destroy the code's spatial content, keep everything else.
                Dn_shuf = Dn[np.ix_(*(2 * [rng.permutation(len(cells))]))]

                r = {'condition': cond, 'hd_mode': hd, 'seed': s, 'n_cells': len(cells),
                     'stress_X': round(stress(Dn, d_x), 4),
                     'stress_XG_c2': round(stress(Dn, d_c2), 4),
                     'stress_XG_c4': round(stress(Dn, d_c4), 4),
                     'stress_sham': round(stress(Dn, d_sham), 4),
                     'stress_shuffled': round(stress(Dn_shuf, d_x), 4),
                     'fold_cos_c2': round(fold_coincidence(M, cells, 'c2'), 4),
                     'fold_cos_c4': round(fold_coincidence(M, cells, 'c4'), 4)}
                r['gain_c2'] = round(r['stress_X'] - r['stress_XG_c2'], 4)
                r['gain_c4'] = round(r['stress_X'] - r['stress_XG_c4'], 4)
                # The artifact-corrected quantity: how much better does the TRUE symmetry do than
                # a sham group of the same order? Only a real fold should prefer the real symmetry.
                r['gain_c2_vs_sham'] = round(r['stress_sham'] - r['stress_XG_c2'], 4)
                rows.append(r)
                print(f"  {cond}/{hd}/s{s:02d}  stress: X={r['stress_X']:.3f}  "
                      f"X/C2={r['stress_XG_c2']:.3f}  sham={r['stress_sham']:.3f}  "
                      f"(shuf {r['stress_shuffled']:.3f})  "
                      f"C2-vs-sham={r['gain_c2_vs_sham']:+.3f}  "
                      f"fold_cos={r['fold_cos_c2']:+.3f}", flush=True)

    if not rows:
        raise SystemExit('no checkpoints found')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} networks)')


if __name__ == '__main__':
    main()
