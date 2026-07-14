"""Do the predictive network's units look like boundary-vector cells?

This is a referee-proofing analysis and we run it expecting an uncomfortable answer.
Uria et al. (2020) showed that an RNN trained on a purely predictive objective from egocentric
vision grows head-direction, boundary-vector and place cells. So the obvious charge against this
paper is: "your network learns BVCs; the quotient law is the boundary-vector-cell model of place
field repetition (Grieves, Duvelle & Dudchenko 2018) with extra steps."

If BVC-like units are there, we should say so, because the honest reading is stronger than the
evasion. A BVC is a MECHANISM: a unit tuned to a boundary at distance d and allocentric bearing
theta. The quotient law is a PRINCIPLE about what any such mechanism can and cannot tell apart.
Note that a BVC's bearing is allocentric, so a BVC population presupposes a compass -- as Julian
et al. (2018) put it, such models "describe how boundaries can be used to calculate one's location
when heading is known, but do not describe how boundaries are used to recover the orientation".
Which is exactly the thing the theorem is about. Finding BVCs would therefore corroborate the
account, not compete with it.

Method. Build a BVC basis over the arena: for each of the four allocentric wall directions and a
range of preferred distances and tuning widths, the idealised firing map of a boundary-vector cell
   f(x) = exp( -(d_theta(x) - d0)^2 / 2 sigma^2 )
where d_theta(x) is the distance from x to the wall lying in direction theta. Fit every unit's rate
map to (a) the best single BVC basis function and (b) the best single 2-D Gaussian place field, and
compare. Both families have the same number of free parameters, so the comparison is fair.

    PYTHONPATH=. python3 analysis/bvc_tuning.py --ckpt-root <dir> --data-root <dir> \
        --out Report/data/bvc_tuning.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

ARENA = 18


def _grid():
    x, y = np.meshgrid(np.arange(1, ARENA + 1), np.arange(1, ARENA + 1), indexing='ij')
    return x.astype(float), y.astype(float)


def bvc_basis():
    """Idealised BVC maps: distance to the wall in each of the four allocentric directions,
    tuned with a Gaussian at preferred distance d0 and width sigma."""
    x, y = _grid()
    # distance from each cell to the wall lying in that allocentric direction
    dist = {'E': ARENA - x, 'W': x - 1.0, 'N': ARENA - y, 'S': y - 1.0}
    maps, meta = [], []
    for th, d in dist.items():
        for d0 in np.arange(0, 13, 1.0):
            for sig in (1.0, 1.75, 2.5, 3.5, 5.0):
                m = np.exp(-((d - d0) ** 2) / (2 * sig ** 2))
                maps.append(m.ravel())
                meta.append((th, float(d0), float(sig)))
    return np.stack(maps), meta


def place_basis():
    """Idealised place fields: a 2-D Gaussian bump at each location, several widths.
    Same functional family and the same number of free parameters as the BVC basis, so the
    model comparison is fair."""
    x, y = _grid()
    maps, meta = [], []
    for cx in range(1, ARENA + 1, 2):
        for cy in range(1, ARENA + 1, 2):
            for sig in (1.5, 2.5, 3.5, 5.0):
                m = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2)))
                maps.append(m.ravel())
                meta.append((cx, cy, float(sig)))
    return np.stack(maps), meta


def _best_corr(R, B, vis):
    """Max Pearson r between each unit's rate map and any basis map, over visited cells."""
    r = R[:, vis]
    b = B[:, vis]
    r = r - r.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    ok_r = (rn[:, 0] > 1e-9)
    C = np.zeros((R.shape[0], B.shape[0]))
    C[ok_r] = (r[ok_r] / rn[ok_r]) @ (b / np.maximum(bn, 1e-9)).T
    return C.max(1), C.argmax(1)


def rate_maps(H, pos):
    U = H.shape[1]
    r = np.zeros((U, ARENA, ARENA))
    occ = np.zeros((ARENA, ARENA))
    xi, yi = pos[:, 0] - 1, pos[:, 1] - 1
    np.add.at(occ, (xi, yi), 1)
    for u in range(U):
        np.add.at(r[u], (xi, yi), H[:, u])
    vis = occ > 0
    r[:, vis] /= occ[vis]
    return r.reshape(U, -1), vis.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--hds', nargs='+', default=['full', 'parity', 'axis', 'const'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--n-traj', type=int, default=600)
    ap.add_argument('--n-states', type=int, default=25_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

    Bb, bmeta = bvc_basis()
    Bp, _ = place_basis()
    print(f'BVC basis {Bb.shape[0]} maps, place basis {Bp.shape[0]} maps', flush=True)

    ds = {}
    for c in a.conds:
        ensure_data(c, a.data_root, a.n_traj, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    for cond in a.conds:
        for hd_mode in a.hds:
            for s in a.seeds:
                p = ck / 'hd_invariance' / cond / hd_mode / f'seed_{s:02d}' / 'ckpt_final.pt'
                if not p.exists():
                    continue
                model = model_from_checkpoint(
                    torch.load(p, map_location='cpu', weights_only=False), dev)
                H, pos, _ = collect_hd(model, ds[cond], hd_mode, a.n_states, dev)
                R, vis = rate_maps(H, pos)

                rb, ib = _best_corr(R, Bb, vis)
                rp, _ = _best_corr(R, Bp, vis)
                # A unit counts as BVC-like if a single boundary-vector basis map explains it well
                # AND does so better than any single place field.
                bvc_like = (rb >= 0.5) & (rb > rp)
                place_like = (rp >= 0.5) & (rp >= rb)
                pref = [bmeta[i][0] for i in ib[bvc_like]] if bvc_like.any() else []

                rows.append({
                    'condition': cond, 'hd_mode': hd_mode, 'seed': s, 'n_units': R.shape[0],
                    'mean_bvc_r': round(float(rb.mean()), 4),
                    'mean_place_r': round(float(rp.mean()), 4),
                    'frac_bvc_like': round(float(bvc_like.mean()), 4),
                    'frac_place_like': round(float(place_like.mean()), 4),
                    'frac_bvc_beats_place': round(float((rb > rp).mean()), 4),
                    'n_bvc_dirs_used': len(set(pref)),
                })
                r = rows[-1]
                print(f"  {cond}/{hd_mode}/s{s:02d}  BVC r={r['mean_bvc_r']:.3f} "
                      f"place r={r['mean_place_r']:.3f}  BVC-like={r['frac_bvc_like']:.2f} "
                      f"place-like={r['frac_place_like']:.2f}", flush=True)

    if not rows:
        raise SystemExit('no checkpoints found')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} networks)')


if __name__ == '__main__':
    main()
