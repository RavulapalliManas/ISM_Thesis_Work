"""PER-UNIT tuning: does the fold live in a subpopulation, or is it a property of the whole code?

WHY THIS EXISTS. Every analysis in this project reports a per-NETWORK mean. That means we have never
been able to ask a question about heterogeneity -- whether all units fold together, whether the fold
is carried by the boundary-tuned units or the place-tuned ones, whether the units that mix most are
the units that fold most. Those are the questions an experimentalist asks first, and we could not
answer any of them. This writes the per-unit table.

THE QUESTION, and it is a real one with a real answer either way. A pure boundary-vector account of
repetition says the fold IS boundary tuning seen twice: the units that repeat should be the
boundary-tuned ones. The quotient law says nothing of the sort -- it says the fold is a property of
the INFORMATION STRUCTURE of the input, so it should be indifferent to what any individual unit
happens to code. So:

    if the fold is carried BY the boundary units  -> the BVC account is doing the work
    if the fold is uniform across cell classes    -> it is a property of the code, not the cells

We do not know which, and the measurement can return either. That is the point.

WHAT IT COMPUTES, one row per unit:
    spatial_info      Skaggs information -- place-ness
    border_frac       fraction of rate-map mass in the outer ring -- border-ness
    m1, m2            harmonics of its direction tuning -- uni/bidirectionality
    mixed             ev_conj - ev_add for THIS unit -- nonlinear mixed selectivity
    sym_c2            correlation of its rate map with its own 180-degree rotation -- ITS fold
    peak_rate, mean_rate

THE READOUT IS `sym_c2`, AND IT IS CONFOUNDED, WHICH WE KNOW BEFORE WE START. An invariant compass
drives sym_c2 to 0.61 in the C1 arena, where nothing can fold (Results). So the per-unit fold score is
NOT read against zero. It is read against ITS OWN C1 BASELINE, network by network, encoding by
encoding, and the quantity of interest is the ARENA-DRIVEN EXCESS. That is the only way this number
means anything, and it is why s1 must be run even though nothing folds there.

PREDICTION, stated before the run. If the quotient law is right and the BVC account is not, then
within a folded network the per-unit fold score should be roughly FLAT across cell classes: the
correlation between border_frac and sym_c2 should be small, and the correlation between spatial_info
and sym_c2 should be small. If instead the fold is concentrated in the boundary units, the first
correlation will be large and positive, and the BVC objection is doing more work than we have allowed.

--------------------------------------------------------------------------------------------------
MEMORY. This is the user's machine, not a cluster, and a job that eats the RAM takes the desktop with
it. Peak footprint, stated in numbers rather than hoped for:

    H            n_states x n_units x 4 bytes  =  20000 x 500 x 4   =  40 MB
    per-hd maps  4 x 324 x n_units x 8 bytes   =  4 x 324 x 500 x 8 =   5 MB
    rows         112 networks x 500 units      =  56000 short dicts =  ~20 MB

One checkpoint is held at a time and freed before the next is loaded; nothing accumulates across the
sweep except the output rows. Peak stays under ~200 MB, an order of magnitude inside the 2 GB budget.
`--n-states` caps the only quantity that scales.

    PYTHONPATH=. python3 analysis/unit_heterogeneity.py \
        --ckpt-root <dir> --data-root <dir> --out Report/data/unit_heterogeneity.csv
"""
from __future__ import annotations
import argparse, csv, gc, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import rot180, ARENA  # noqa: E402
from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.analysis.cell_properties import _ev, _ev_additive, _skaggs  # noqa: E402
from project5_symmetry.analysis.compass_symmetry import harmonics, _tuning  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

N_HD = 4


def rate_maps(H, cx):
    """[U, ARENA*ARENA] occupancy-normalised rate maps, plus the visited mask."""
    U = H.shape[1]
    ncell = ARENA * ARENA
    sums = np.zeros((ncell, U), dtype=np.float64)
    counts = np.zeros(ncell)
    np.add.at(sums, cx, H)
    np.add.at(counts, cx, 1)
    vis = counts > 0
    maps = np.zeros((U, ncell))
    maps[:, vis] = (sums[vis] / counts[vis, None]).T
    return maps, vis, counts


def sym_c2(maps, vis):
    """Per-unit correlation of a rate map with its own 180-degree rotation.

    THIS IS NOT A FOLD SCORE ON ITS OWN. An invariant compass drives it to 0.61 in the C1 arena where
    nothing can fold. It is only interpretable against its own C1 baseline -- which is why s1 is in
    the sweep.
    """
    idx = np.arange(ARENA * ARENA)
    xy = np.stack([idx // ARENA + 1, idx % ARENA + 1], 1)
    rot = rot180(xy, ARENA)
    ridx = (rot[:, 0] - 1) * ARENA + (rot[:, 1] - 1)
    both = vis & vis[ridx]
    if both.sum() < 4:
        return np.zeros(maps.shape[0])
    A, B = maps[:, both], maps[:, ridx[both]]
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    return np.divide((A * B).sum(1), den, out=np.zeros(len(A)), where=den > 1e-12)


def border_frac(maps, vis):
    """Fraction of a unit's rate-map mass in the outer two-cell ring."""
    idx = np.arange(ARENA * ARENA)
    r, c = idx // ARENA, idx % ARENA
    ring = (r < 2) | (r >= ARENA - 2) | (c < 2) | (c >= ARENA - 2)
    tot = maps[:, vis].sum(1)
    out = maps[:, ring & vis].sum(1)
    return np.divide(out, tot, out=np.zeros(len(maps)), where=tot > 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--encodings', nargs='+', default=['full', 'axis', 'parity', 'const'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--n-states', type=int, default=20_000,
                    help='caps the only quantity that scales with memory')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

    ds = {}
    for c in a.conds:
        ensure_data(c, a.data_root, 800, a.threads, size=18)
        ds[c] = TrajectoryDataset(str(Path(a.data_root) / c))

    rows = []
    for cond in a.conds:
        for enc in a.encodings:
            for s in a.seeds:
                p = ck / 'hd_invariance' / cond / enc / f'seed_{s:02d}' / 'ckpt_final.pt'
                if not p.exists():
                    continue
                model = model_from_checkpoint(
                    torch.load(p, map_location='cpu', weights_only=False), dev)
                with torch.no_grad():
                    H, pos, hd = collect_hd(model, ds[cond], enc, a.n_states, dev, seed=s)
                cx = (pos[:, 0] - 1) * ARENA + (pos[:, 1] - 1)

                maps, vis, counts = rate_maps(H, cx)
                occ = counts[vis] / counts[vis].sum()
                si = _skaggs(maps[:, vis], occ)
                bf = border_frac(maps, vis)
                fold = sym_c2(maps, vis)
                T, _ = _tuning(H, cx, hd)
                m1, m2, _, _ = harmonics(T)
                # per-unit mixed selectivity: the conjunctive model beyond the additive one
                cc = cx * N_HD + hd.astype(np.int64)
                mixed = np.maximum(_ev(H, cc) - _ev_additive(H, cx, hd.astype(np.int64)), 0.0)

                for u in range(H.shape[1]):
                    rows.append({
                        'condition': cond, 'encoding': enc, 'seed': s, 'unit': u,
                        'spatial_info': round(float(si[u]), 4),
                        'border_frac': round(float(bf[u]), 4),
                        'sym_c2': round(float(fold[u]), 4),
                        'm1': round(float(m1[u]), 4), 'm2': round(float(m2[u]), 4),
                        'mixed': round(float(mixed[u]), 4),
                        'mean_rate': round(float(H[:, u].mean()), 4),
                        'peak_rate': round(float(maps[u, vis].max()), 4),
                    })
                print(f"  {cond}/{enc:<6s}/s{s:02d}  {H.shape[1]} units  "
                      f"sym_c2={fold.mean():.3f}  SI={si.mean():.3f}  mixed={mixed.mean():.3f}",
                      flush=True)
                # 6.5: free as you go. One checkpoint in memory at a time, never the sweep.
                del H, pos, hd, maps, model
                gc.collect()

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} units)')


if __name__ == '__main__':
    main()
