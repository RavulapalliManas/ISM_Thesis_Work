"""Real-data test of the HD-invariance prediction in Hockeimer et al. (2023) CA1 city-block data.

Data: Hockeimer, Lai, Natrajan, Knierim (eLife 2023), Johns Hopkins Research Data Repository,
doi:10.7281/T15HMQD4, file `AlleySuperpopDirVisitFiltered.csv` (one row per visit to a field's
alley: firing Rate, travel direction CurrDir, alley Orientation V/H, Repeating, etc.).

Prediction (from our model): two repeating fields of one cell in alleys related by TRANSLATION
(same orientation -> same travel axis -> the head-direction signal reads the same) are
HD-indistinguishable, so the code folds them onto one representation and they should carry the
SAME directional preference. Fields in alleys related by ROTATION (different orientation,
orthogonal travel axes) lift apart.

Readouts (fold signature = same-orientation repeating fields share directional preference):
  - directional index DI = (rate[+]-rate[-])/(rate[+]+rate[-]); +/- = E/W (H alleys), N/S (V).
  - within-cell same-orientation field-pair DI correlation vs a shuffle null.
  - mixed-effects ICC of DI with rat and cell:orientation random effects.

    python3 hockeimer_reanalysis.py --data <dir with AlleySuperpopDirVisitFiltered.csv> \
        --out project5_symmetry/Report/data/hockeimer_field_di.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

MIN_VISITS = 3


def field_di(df):
    rows = []
    for (rat, day, cid, fid), g in df.groupby(['Rat', 'Day', 'CellID', 'FieldID']):
        orient = g.Orientation.iloc[0]
        pos, neg = ('E', 'W') if orient == 'H' else ('N', 'S')
        rp, rn = g[g.CurrDir == pos].Rate, g[g.CurrDir == neg].Rate
        if len(rp) < MIN_VISITS or len(rn) < MIN_VISITS or (rp.mean() + rn.mean()) <= 1e-9:
            continue
        rows.append({'rat': rat, 'cell': f'{rat}_{day}_{cid}', 'fid': int(fid), 'orient': orient,
                     'rep': bool(g.Repeating.iloc[0]),
                     'di': (rp.mean() - rn.mean()) / (rp.mean() + rn.mean())})
    return pd.DataFrame(rows)


def paired_di(fd):
    xs, ys = [], []
    for _, g in fd[fd.rep].groupby('cell'):
        for _, go in g.groupby('orient'):
            di = go.di.values
            for i in range(len(di)):
                for j in range(len(di)):
                    if i != j:
                        xs.append(di[i]); ys.append(di[j])
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dir with AlleySuperpopDirVisitFiltered.csv')
    ap.add_argument('--out', required=True, help='per-field DI table (csv)')
    ap.add_argument('--shuffles', type=int, default=2000)
    a = ap.parse_args()

    df = pd.read_csv(Path(a.data) / 'AlleySuperpopDirVisitFiltered.csv')
    df = df[df.Traversal == True]
    fd = field_di(df)
    print(f'fields with stable DI: {len(fd)}  (repeating: {int(fd.rep.sum())})')

    x, y = paired_di(fd)
    r, p = pearsonr(x, y)
    print(f'\nSame-orientation repeating field pairs: {len(x)//2}')
    print(f'  DI correlation r = {r:+.3f}   p = {p:.3g}')

    rng = np.random.default_rng(0)
    fdr, other = fd[fd.rep].copy(), fd[~fd.rep]
    null = []
    for _ in range(a.shuffles):
        s = fdr.copy()
        s['di'] = s.groupby('orient').di.transform(lambda v: rng.permutation(v.values))
        xs, ys = paired_di(pd.concat([s, other]))
        null.append(pearsonr(xs, ys)[0] if len(xs) > 2 else 0.0)
    null = np.array(null)
    p_shuf = (np.sum(null >= r) + 1) / (len(null) + 1)
    print(f'  shuffle null r = {null.mean():+.3f} +/- {null.std():.3f};  p = {p_shuf:.4f}')

    import statsmodels.formula.api as smf
    m = fd[fd.rep].copy()
    m['co'] = m.cell + '_' + m.orient
    m = m[m.groupby('co').di.transform('count') >= 2]
    icc = float('nan')
    try:
        md = smf.mixedlm('di ~ 1', m, groups=m['rat'],
                         vc_formula={'cell': '0 + C(co)'}).fit(reml=True, method='lbfgs')
        vc, resid = float(md.vcomp[0]), float(md.scale)
        icc = vc / (vc + resid)
        print(f'\nMixed model DI ~ 1 + (1|rat) + (1|cell:orient): n={len(m)} fields, '
              f'{m.rat.nunique()} rats, cell:orient var={vc:.4f}, resid={resid:.4f}, ICC={icc:.3f}')
    except Exception as e:
        print(f'\nmixed model failed: {e}')

    span = fd[fd.rep].groupby('cell').orient.nunique()
    print(f'\nRepeating cells spanning both orientations (rotation-related fields): '
          f'{int((span >= 2).sum())}/{span.size}')

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fd.to_csv(a.out, index=False)
    summ = Path(a.out).with_name('hockeimer_summary.csv')
    pd.DataFrame([{'n_fields': len(fd), 'n_repeating': int(fd.rep.sum()),
                   'n_same_orient_pairs': len(x) // 2, 'pair_r': r, 'pair_p': p,
                   'shuffle_p': p_shuf, 'mixed_icc': icc,
                   'cells_span_both_orient': int((span >= 2).sum())}]).to_csv(summ, index=False)
    print(f'\nwrote {a.out} and {summ.name}')


if __name__ == '__main__':
    main()
