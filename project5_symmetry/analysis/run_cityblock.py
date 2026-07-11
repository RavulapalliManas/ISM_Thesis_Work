"""Model-side directional-repetition readout on the city-block maze, mirroring the Hockeimer
reanalysis, for model-data correspondence.

For each trained unit we form a directional index (DI) per alley SEGMENT it fires in
(rate[dir+]-rate[dir-])/(sum); dir+/- = E/W for horizontal segments, N/S for vertical. The
model predicts that a unit's fields in same-orientation (translation-related) segments share
directional preference -- so within-unit, same-orientation segment-DI pairs should correlate
positively (fold), matching the r/ICC measured in the real CA1 data.

    python3 run_cityblock.py --runs <runs/multi/cityblock> --data-root <data/cityblock> --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.cell_types import collect_hd  # noqa: E402
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.cityblock import ARENA, GRID_LINES, cell_orientation  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

# heading code (MiniGrid) 0=E 1=S 2=W 3=N ; +/- along each axis
POS_NEG = {'H': (0, 2), 'V': (3, 1)}


def segments():
    segs = []
    for r in GRID_LINES:
        run = []
        for c in range(1, ARENA + 1):
            if cell_orientation(r, c) == 'H':
                run.append((r, c))
            elif run:
                segs.append(('H', run)); run = []
        if run:
            segs.append(('H', run))
    for c in GRID_LINES:
        run = []
        for r in range(1, ARENA + 1):
            if cell_orientation(r, c) == 'V':
                run.append((r, c))
            elif run:
                segs.append(('V', run)); run = []
        if run:
            segs.append(('V', run))
    return segs


def unit_field_dis(hidden, pos, head, segs, min_visits=5, field_frac=0.5):
    """For one unit: list of (orient, DI) over segments it fires in."""
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(pos[:, 0], pos[:, 1]))}  # unused
    out = []
    seg_rate = []
    per = []
    for orient, cells in segs:
        cs = set(cells)
        mask = np.array([(int(x), int(y)) in cs for x, y in pos])
        if mask.sum() < 2 * min_visits:
            per.append(None); continue
        pos_h, neg_h = POS_NEG[orient]
        rp = hidden[mask & (head == pos_h)]
        rn = hidden[mask & (head == neg_h)]
        if len(rp) < min_visits or len(rn) < min_visits:
            per.append(None); continue
        mp, mn = rp.mean(), rn.mean()
        per.append((orient, mp, mn, (mp + mn) / 2))
        seg_rate.append((mp + mn) / 2)
    if not seg_rate:
        return out
    thr = field_frac * max(seg_rate)
    for p in per:
        if p is None:
            continue
        orient, mp, mn, r = p
        if r >= thr and (mp + mn) > 1e-9:
            out.append((orient, (mp - mn) / (mp + mn)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    segs = segments()
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints; {sum(o=="H" for o,_ in segs)} H + '
          f'{sum(o=="V" for o,_ in segs)} V segments', flush=True)
    ds = TrajectoryDataset(str(Path(a.data_root) / 'lattice'))

    rows = []                       # (seed, unit, orient, di)
    for p in ckpts:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        seed = ck['meta']['seed']
        model = model_from_checkpoint(ck, dev)
        H, pos, head = collect_hd(model, ds, 'full', a.n_states, dev)
        for u in range(H.shape[1]):
            for orient, di in unit_field_dis(H[:, u], pos, head, segs):
                rows.append({'seed': seed, 'unit': u, 'orient': orient, 'di': di})
        print(f'  seed{seed}: {sum(r["seed"]==seed for r in rows)} field-DIs', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'unit', 'orient', 'di']); w.writeheader(); w.writerows(rows)

    # same-orientation within-unit field-pair DI correlation
    from scipy.stats import pearsonr
    import pandas as pd
    df = pd.DataFrame(rows)
    xs, ys = [], []
    for (_, _), g in df.groupby(['seed', 'unit']):
        for _, go in g.groupby('orient'):
            d = go.di.values
            for i in range(len(d)):
                for j in range(len(d)):
                    if i != j:
                        xs.append(d[i]); ys.append(d[j])
    if len(xs) > 2:
        r, pp = pearsonr(xs, ys)
        print(f'\nsame-orientation field-DI pairs: {len(xs)//2}  r = {r:+.3f}  p = {pp:.3g}')
        print('(compare to real CA1: r=+0.23, ICC=0.31)')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
