"""Single-cell properties across arenas (symmetry) and head-direction encodings (information and
invariance): the panel an experimentalist would compute from a tetrode recording.

Per unit:
  spatial_info   Skaggs information, bits/spike:  sum_x p(x) (r(x)/rbar) log2(r(x)/rbar)
  sparsity       Skaggs sparsity: (sum p r)^2 / sum p r^2. Low = compact code.
  selectivity    peak rate / mean rate
  coherence      Muller-Kubie: corr(bin rate, mean of its 8 neighbours), z-transformed
  n_fields       contiguous regions above 50% of peak, >= 3 cells
  field_area     mean area of those fields
  ev_pos         variance of the unit explained by position alone
  ev_hd          ... by head direction alone
  ev_add         ... by the best ADDITIVE model f(x) + g(h)
  ev_conj        ... by the full conjunctive model f(x, h)
  mixed          ev_conj - ev_add:  NONLINEAR MIXED SELECTIVITY, the variance that position and
                 head direction explain jointly but neither explains additively. This is the
                 conjunctive place-by-direction coding a folded code is often confused with.
  hd_rayleigh    resultant vector length over the four headings

Why this matters for the paper. Two lesion studies disagree about what removing head direction
does to place cells. Harland et al. (2017), in identical compartments with no polarizing cue,
report LOWER spatial information (1.29 -> 1.05 bits) and HIGHER sparsity (0.32 -> 0.37). Calton
et al. (2003), in a cue-controlled cylinder, report NO change in field number (1.29 -> 1.53, n.s.),
field size, rate or sparsity. The quotient law says the difference is the symmetry of the
environment, not the lesion: a compass can only fold a map onto a symmetry that exists. Ablating
the compass across the C1/C2/C4 arenas tests exactly that, and separates the part of the effect
that is degradation (present even in C1, where nothing folds) from the part that is folding.

    PYTHONPATH=. python3 analysis/cell_properties.py --ckpt-root <dir> --data-root <dir> \
        --out Report/data/cell_properties.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.field_stats import count_fields  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import ensure_data  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

ARENA = 18
N_HD = 4


def _maps(H, pos, arena=ARENA):
    """(rate maps [U, A, A], occupancy [A, A], visited mask)."""
    U = H.shape[1]
    r = np.zeros((U, arena, arena))
    occ = np.zeros((arena, arena))
    xi, yi = pos[:, 0] - 1, pos[:, 1] - 1
    np.add.at(occ, (xi, yi), 1)
    for u in range(U):
        np.add.at(r[u], (xi, yi), H[:, u])
    vis = occ > 0
    r[:, vis] /= occ[vis]
    return r, occ, vis


def _ev(h, codes):
    """Fraction of a unit's variance explained by a lookup table indexed by `codes`."""
    order = np.argsort(codes, kind='stable')
    cs, hs = codes[order], h[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    pred = np.empty_like(hs)
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
        pred[a:b] = hs[a:b].mean(axis=0)
    tot = hs.var(axis=0)
    res = (hs - pred).var(axis=0)
    return np.where(tot > 1e-12, 1.0 - res / np.maximum(tot, 1e-12), 0.0)


def _ev_additive(h, cx, cd):
    """EV of the best additive model f(x) + g(h), fitted by alternating means (2 passes suffice
    for a balanced-enough design; this is a lower bound on the additive fit, so `mixed` is
    conservative)."""
    fx = np.zeros_like(h)
    for _ in range(3):
        resid = h - fx
        gd = _lookup(resid, cd)
        resid2 = h - gd
        fx = _lookup(resid2, cx)
    pred = fx + _lookup(h - fx, cd)
    tot = h.var(axis=0)
    res = (h - pred).var(axis=0)
    return np.where(tot > 1e-12, 1.0 - res / np.maximum(tot, 1e-12), 0.0)


def _lookup(h, codes):
    order = np.argsort(codes, kind='stable')
    cs, hs = codes[order], h[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    pred = np.empty_like(hs)
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cs)]):
        pred[a:b] = hs[a:b].mean(axis=0)
    out = np.empty_like(h)
    out[order] = pred
    return out


def _coherence(m, vis):
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], float)
    from scipy.ndimage import convolve
    s = convolve(np.where(vis, m, 0.0), k, mode='constant')
    n = convolve(vis.astype(float), k, mode='constant')
    nb = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    a, b = m[vis], nb[vis]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    r = min(max(r, -0.999), 0.999)
    return float(np.arctanh(r))


def _skaggs(rate, p):
    """Skaggs information in bits per unit of activity: sum_x p(x) (r/rbar) log2(r/rbar).
    `rate` is (U, nbins), `p` the occupancy over those bins."""
    rbar = (rate * p).sum(1)
    ok = rbar > 1e-9
    ratio = np.divide(rate, rbar[:, None], out=np.zeros_like(rate), where=ok[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(p * ratio * np.log2(np.where(ratio > 0, ratio, 1.0)), axis=1)


def properties(H, pos, hd):
    r, occ, vis = _maps(H, pos)
    p = occ[vis] / occ[vis].sum()
    U = H.shape[1]

    rm = r[:, vis]                                   # [U, ncells]
    rbar = (rm * p).sum(1)                           # occupancy-weighted mean rate
    ok = rbar > 1e-9
    ratio = np.divide(rm, rbar[:, None], out=np.zeros_like(rm), where=ok[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        si = np.nansum(p * ratio * np.log2(np.where(ratio > 0, ratio, 1.0)), axis=1)
    sparsity = np.divide((rm * p).sum(1) ** 2, ((rm ** 2) * p).sum(1),
                         out=np.ones(U), where=((rm ** 2) * p).sum(1) > 1e-12)
    selectivity = np.divide(rm.max(1), rbar, out=np.zeros(U), where=ok)

    cx = (pos[:, 0] - 1) * ARENA + (pos[:, 1] - 1)
    cd = hd.astype(np.int64)
    cc = cx * N_HD + cd
    ev_pos, ev_hd = _ev(H, cx), _ev(H, cd)
    ev_conj = _ev(H, cc)
    ev_add = _ev_additive(H, cx, cd)
    mixed = np.maximum(ev_conj - ev_add, 0.0)

    hdm = np.stack([H[hd == d].mean(0) if (hd == d).any() else np.zeros(U)
                    for d in range(N_HD)], 1)        # [U, 4]
    ang = 2 * np.pi * np.arange(N_HD) / N_HD
    tot = hdm.sum(1)
    rayl = np.divide(np.abs((hdm * np.exp(1j * ang)).sum(1)), tot,
                     out=np.zeros(U), where=tot > 1e-9)

    # --- rates. Both lesion studies report firing rates UNCHANGED, so if ours move, that is a
    # real mismatch and we need to know it.
    mean_rate = H.mean(0)
    peak_rate = rm.max(1)

    # --- directional information, computed TWO ways, because it is not yet established which one
    # Calton et al. used and the two make opposite predictions for us.
    #   MARGINAL: Skaggs information over heading, ignoring position. Pure directional tuning.
    #             We expect this to COLLAPSE when the compass is removed.
    #   CONJUNCTIVE: the same, but computed within each position bin and averaged over positions,
    #             i.e. how much a cell's rate depends on heading GIVEN where the animal is. We
    #             expect this to RISE, because heading then enters only through the egocentric view,
    #             which binds it to position.
    p_hd = np.array([(hd == d).mean() for d in range(N_HD)])
    dir_info_marginal = _skaggs(hdm, p_hd)

    # IN-FIELD directional information -- Calton et al.'s (2003) measure, verified from their
    # Methods: Skaggs information over head-direction bins, "calculated only for cell activity when
    # the animal was within the place field of the recorded cell". It is therefore a CONJUNCTIVE
    # quantity (is this cell's field active only for some headings?), not marginal HD tuning. Their
    # place field is "the largest contiguous group of pixels possessing a firing rate >= 10% of the
    # average firing rate of the three highest firing rate pixels", 4-connectivity.
    from scipy.ndimage import label
    xi_all, yi_all = pos[:, 0] - 1, pos[:, 1] - 1
    dir_info_field = np.zeros(U)
    for u in range(U):
        m = r[u]
        flat = np.sort(m[vis])[::-1]
        if flat.size < 3 or flat[0] <= 0:
            continue
        thr = 0.10 * flat[:3].mean()
        binm = (m >= thr) & vis
        lab, n = label(binm)                          # 4-connectivity, as they specify
        if n == 0:
            continue
        sizes = [(lab == i).sum() for i in range(1, n + 1)]
        field = lab == (1 + int(np.argmax(sizes)))    # "the largest place field was used"
        inf = field[xi_all, yi_all]
        if inf.sum() < 4 * 10:
            continue
        hu, du = H[inf, u], hd[inf]
        tune = np.array([hu[du == d].mean() if (du == d).any() else 0.0 for d in range(N_HD)])
        pin = np.array([(du == d).mean() for d in range(N_HD)])
        dir_info_field[u] = _skaggs(tune[None, :], pin)[0]

    # --- stability. Harland et al. report session-to-session stability falling from 0.620 to
    # 0.347 after the lesion. The model's analogue is a split-half correlation of the rate map:
    # build the map from the first and the second half of the states and correlate them.
    half = len(H) // 2
    r1, _, v1 = _maps(H[:half], pos[:half])
    r2, _, v2 = _maps(H[half:], pos[half:])
    both = v1 & v2
    a1, a2 = r1[:, both], r2[:, both]
    a1 = a1 - a1.mean(1, keepdims=True)
    a2 = a2 - a2.mean(1, keepdims=True)
    den = np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1)
    stability = np.divide((a1 * a2).sum(1), den, out=np.zeros(U), where=den > 1e-9)

    mask = ~vis
    nf, fa, coh = np.zeros(U), np.zeros(U), np.zeros(U)
    for u in range(U):
        nf[u], fa[u] = count_fields(r[u], mask)
        coh[u] = _coherence(r[u], vis)

    place = (si >= 0.3) & (nf >= 1)

    def pm(v):
        return float(np.mean(v[place])) if place.any() else float('nan')

    return dict(
        n_units=U, frac_place=float(place.mean()),
        spatial_info=pm(si), sparsity=pm(sparsity), selectivity=pm(selectivity),
        coherence=pm(coh), n_fields=pm(nf),
        field_area=float(np.mean(fa[place][nf[place] > 0])) if (place & (nf > 0)).any() else np.nan,
        # rates: both lesion studies report these unchanged
        mean_rate=pm(mean_rate), peak_rate=pm(peak_rate),
        # directional information, both ways (see note above). `dir_info_field` is the one that is
        # comparable to Calton et al. (2003): Skaggs over heading, computed on IN-FIELD samples only.
        dir_info_marginal=pm(dir_info_marginal), dir_info_field=pm(dir_info_field),
        # Harland's session-to-session stability; ours is a split-half of the rate map
        stability=pm(stability),
        ev_pos=float(np.mean(ev_pos)), ev_hd=float(np.mean(ev_hd)),
        ev_add=float(np.mean(ev_add)), ev_conj=float(np.mean(ev_conj)),
        mixed=float(np.mean(mixed)), hd_rayleigh=float(np.mean(rayl)),
        frac_mixed=float(np.mean(mixed > 0.05)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--hds', nargs='+', default=['full', 'parity', 'axis', 'const'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--n-traj', type=int, default=800)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ck = Path(a.ckpt_root)

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
                H, pos, hd = collect_hd(model, ds[cond], hd_mode, a.n_states, dev)
                r = {'condition': cond, 'hd_mode': hd_mode, 'seed': s}
                r.update({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in properties(H, pos, hd).items()})
                rows.append(r)
                print(f"  {cond}/{hd_mode}/s{s:02d}  SI={r['spatial_info']:.3f} "
                      f"sparsity={r['sparsity']:.3f} fields={r['n_fields']:.2f} "
                      f"area={r['field_area']:.1f} mixed={r['mixed']:.3f}", flush=True)

    if not rows:
        raise SystemExit('no checkpoints found')
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} networks)')


if __name__ == '__main__':
    main()
