"""The three numbers in the manuscript that no code in this repository produced.

WHY THIS EXISTS. An audit found that the paper states a Fano factor, a gridness score and an
omnidirectionality index -- with tests attached -- and that NO SCRIPT ANYWHERE IN THE REPO COMPUTES
ANY OF THEM. They were not lost in a transfer from the cloud: a search of the whole tree found the
words nowhere outside the manuscript. A number with no provenance is a fabrication until proven
otherwise, so this file either reproduces them or convicts them.

THE THREE CLAIMS, quoted from the manuscript, to be confirmed or corrected:

    Fano factor        "field counts are dispersed (Fano factor 0.52; chi^2 against a Poisson of the
                       same mean is decisively rejected)"
    gridness           "gridness over the 100 most spatially tuned units: mean -0.35, maximum +0.08,
                       none exceeding 0.08" -- i.e. NO GRID CELLS, which the paper leans on when it
                       says Gardner's toroidal topology cannot exist in this model
    omnidirectionality "mean 0.56, median 0.64, 35% of unit pairs above 0.8"

MEMORY. Per-unit rate maps only; one checkpoint at a time, freed before the next.
    H          20000 x 500 x 4 B = 40 MB
    maps       500 x 324 x 8 B   =  1 MB
    per-hd     4 x 500 x 324 x 8 =  5 MB
Peak well under 200 MB.

    PYTHONPATH=. python3 analysis/orphan_metrics.py --ckpt-root <dir> --data-root <dir> \
        --out Report/data/orphan_metrics.csv
"""
from __future__ import annotations
import argparse, csv, gc, math, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chisquare
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.run_phase_decoding import ARENA  # noqa: E402
from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.analysis.cell_properties import _skaggs  # noqa: E402
from project5_symmetry.analysis.unit_heterogeneity import rate_maps  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

N_HD = 4


def n_fields(m2d):
    """Contiguous regions above 10% of the peak (Calton et al.'s criterion, 4-connectivity)."""
    thr = 0.1 * m2d.max()
    if thr <= 0:
        return 0
    lab, n = label(m2d >= thr, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return int(sum((lab == i).sum() >= 2 for i in range(1, n + 1)))


def gridness(m2d):
    """Standard gridness: the 60-degree minus 45-degree contrast of the rate-map autocorrelogram.

    A hexagonal lattice peaks at 60 and 120 degrees and troughs at 30, 90 and 150. Gridness is the
    minimum of the 60/120 correlations minus the maximum of the 30/90/150 ones. A grid cell scores
    well above 0; anything at or below 0 is not a grid.
    """
    m = m2d - m2d.mean()
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(m, s=(2 * ARENA, 2 * ARENA))) ** 2)))
    ac /= ac.max() + 1e-12
    c = np.array(ac.shape) // 2
    yy, xx = np.mgrid[:ac.shape[0], :ac.shape[1]]
    rr = np.hypot(yy - c[0], xx - c[1])
    ring = (rr > 2) & (rr < ARENA)                       # exclude the central peak
    if ring.sum() < 20:
        return 0.0

    def rot_corr(deg):
        th = np.deg2rad(deg)
        ys = (yy - c[0]) * np.cos(th) - (xx - c[1]) * np.sin(th) + c[0]
        xs = (yy - c[0]) * np.sin(th) + (xx - c[1]) * np.cos(th) + c[1]
        yi = np.clip(np.round(ys).astype(int), 0, ac.shape[0] - 1)
        xi = np.clip(np.round(xs).astype(int), 0, ac.shape[1] - 1)
        a, b = ac[ring], ac[yi[ring], xi[ring]]
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / d) if d > 1e-12 else 0.0

    return min(rot_corr(60), rot_corr(120)) - max(rot_corr(30), rot_corr(90), rot_corr(150))


def omnidirectional(H, cx, hd, vis):
    """Per-unit: mean correlation between its rate maps computed at DIFFERENT headings.

    A cell that fires at the same place regardless of which way the animal faces is omnidirectional
    (real place cells largely are). A cell whose field depends on heading is not. We take the mean
    over the six heading PAIRS.
    """
    U = H.shape[1]
    maps = np.zeros((N_HD, U, ARENA * ARENA))
    for d in range(N_HD):
        sel = hd == d
        if sel.sum() < 50:
            return np.zeros(U), np.zeros(0)
        m, v, _ = rate_maps(H[sel], cx[sel])
        maps[d] = m
    both = vis.copy()
    pairs = []
    for i in range(N_HD):
        for j in range(i + 1, N_HD):
            A, B = maps[i][:, both], maps[j][:, both]
            A = A - A.mean(1, keepdims=True)
            B = B - B.mean(1, keepdims=True)
            d = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
            pairs.append(np.divide((A * B).sum(1), d, out=np.zeros(U), where=d > 1e-12))
    P = np.stack(pairs, 1)                                # [U, 6]
    return P.mean(1), P.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--encodings', nargs='+', default=['full', 'axis', 'parity', 'const'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--n-states', type=int, default=20_000)
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

                m2d = maps.reshape(-1, ARENA, ARENA)
                nf = np.array([n_fields(m) for m in m2d])
                # FANO of the field-count distribution across units, and a Poisson test.
                mu = nf.mean()
                fano = float(nf.var() / mu) if mu > 1e-9 else 0.0
                mx = int(nf.max())
                obs = np.bincount(nf, minlength=mx + 1).astype(float)
                lam = mu
                exp = np.array([np.exp(-lam) * lam ** k / math.factorial(k)
                                for k in range(mx + 1)]) * len(nf)
                keep = exp > 5
                chi2 = chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum())

                # GRIDNESS on the 100 most spatially tuned units -- the paper's own criterion.
                top = np.argsort(si)[-100:]
                g = np.array([gridness(m2d[u]) for u in top])

                omn, allpairs = omnidirectional(H, cx, hd, vis)

                rows.append({
                    'condition': cond, 'encoding': enc, 'seed': s, 'n_units': int(H.shape[1]),
                    'fano_fields': round(fano, 4), 'mean_fields': round(float(mu), 4),
                    'chi2_poisson': round(float(chi2.statistic), 2),
                    'chi2_p': f'{chi2.pvalue:.3e}',
                    'gridness_mean': round(float(g.mean()), 4),
                    'gridness_max': round(float(g.max()), 4),
                    'gridness_frac_above_0p08': round(float((g > 0.08).mean()), 4),
                    'omni_mean': round(float(omn.mean()), 4),
                    'omni_median': round(float(np.median(omn)), 4),
                    'omni_frac_pairs_above_0p8': round(float((allpairs > 0.8).mean()), 4),
                })
                print(f"  {cond}/{enc:<6s}/s{s:02d}  fano={fano:.3f}  grid mean={g.mean():+.3f} "
                      f"max={g.max():+.3f}  omni med={np.median(omn):.3f}", flush=True)
                del H, pos, hd, maps, m2d, model
                gc.collect()

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
