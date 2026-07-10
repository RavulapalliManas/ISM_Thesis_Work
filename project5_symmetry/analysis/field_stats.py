"""Place-field statistics per unit: field count, size, peak, spatial information.

The headline quantity is the NUMBER OF FIELDS PER UNIT. Folding predicts it exactly: when the
head-direction encoding is invariant to the arena's symmetry group G, one place field is
reported at every one of the |G| orbit-mates, so a place cell should carry |G| fields. This
turns the rate-map picture into a number and replicates Grieves et al. (2016), who found
multi-field firing increasing with the rotational symmetry of the arena.

A field is a connected region of the smoothed rate map at or above half its peak, of at least
`min_area` cells. A unit is counted as a place cell if it has at least one such field and its
Skaggs spatial information exceeds `si_thresh`.

    python3 field_stats.py --runs /root/runs/hd_invariance --conds s1 s2 s4 \
        --out /root/results8/field_stats.csv --dump /root/results8/field_counts.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.rate_maps import mean_rate_maps, spatial_information  # noqa: E402
from project5_symmetry.analysis.run_spectrum import (collect, identify,  # noqa: E402
                                                     model_from_checkpoint)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

HD_MODES = ('full', 'parity', 'axis', 'const')


def _smooth(m, mask, sigma=0.9):
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(mask, 0.0, m), sigma)
    den = gaussian_filter((~mask).astype(float), sigma)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)


def count_fields(m, mask, peak_frac=0.5, min_area=3):
    """(n_fields, mean_field_area) for one smoothed rate map."""
    from scipy.ndimage import label
    ms = _smooth(m, mask)
    peak = ms.max()
    if peak <= 0:
        return 0, 0.0
    binm = (ms >= peak_frac * peak) & (~mask)
    lab, n = label(binm)
    areas = [int((lab == i).sum()) for i in range(1, n + 1)]
    fields = [a for a in areas if a >= min_area]
    return len(fields), (float(np.mean(fields)) if fields else 0.0)


def rot_symmetry(m, mask, order):
    """Rotational self-similarity of a rate map: correlation with its own rotations under the
    group. This is the FOLD-SPECIFIC statistic. Unlike raw field count, it is not confounded
    by the encoding, because extra fields raise the count everywhere but only fields placed at
    group-related positions raise this. It reads high only when a map actually carries the
    arena's symmetry."""
    a = _smooth(m, mask)
    a = (a - a.mean()).ravel()
    if a.std() < 1e-9:
        return 0.0
    js = (2,) if order == 2 else (1, 2, 3)
    cs = [np.corrcoef(a, (np.rot90(_smooth(m, mask), j) - _smooth(m, mask).mean()).ravel())[0, 1]
          for j in js]
    return float(np.mean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--conds', nargs='+', default=['s1', 's2', 's4'])
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--si-thresh', type=float, default=0.3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dump', default=None, help='npz of per-unit field counts for plotting')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    ckpts = [p for p in ckpts if identify(p, torch.load(p, map_location='cpu',
             weights_only=False)['meta'])[0] in a.conds]
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt for {a.conds} under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    datasets, rows, dump = {}, [], {}
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        hidden, pos = collect(model, datasets[cond], hd, a.n_states, dev)
        maps, occ = mean_rate_maps(hidden, pos, a.arena)
        mask = occ == 0
        si = spatial_information(maps, occ)
        counts, sizes, sym2, sym4 = [], [], [], []
        for u in range(maps.shape[0]):
            if si[u] < a.si_thresh:
                continue
            nf, fs = count_fields(maps[u], mask)
            if nf >= 1:
                counts.append(nf); sizes.append(fs)
                sym2.append(rot_symmetry(maps[u], mask, 2))
                sym4.append(rot_symmetry(maps[u], mask, 4))
        counts = np.array(counts)
        n_place = len(counts)
        rows.append({'condition': cond, 'hd_mode': hd, 'seed': seed, 'n_place': n_place,
                     'mean_fields': float(counts.mean()) if n_place else float('nan'),
                     'median_fields': float(np.median(counts)) if n_place else float('nan'),
                     'frac_multifield': float((counts >= 2).mean()) if n_place else float('nan'),
                     'mean_field_area': float(np.mean(sizes)) if sizes else float('nan'),
                     'sym_c2': float(np.mean(sym2)) if sym2 else float('nan'),
                     'sym_c4': float(np.mean(sym4)) if sym4 else float('nan')})
        dump[f'{cond}_{hd}_s{seed}'] = counts.astype(np.int16)
        r = rows[-1]
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/seed{seed}  n_place={n_place:3d}  '
              f'mean_fields={r["mean_fields"]:.2f}  frac>=2={r["frac_multifield"]:.2f}  '
              f'symC2={r["sym_c2"]:+.2f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.dump:
        np.savez_compressed(a.dump, **dump)

    print(f'\n{"cond/hd":<12s} {"n_place":>8s} {"mean_fields":>12s} {"frac>=2":>9s} '
          f'{"sym_c2":>8s} {"sym_c4":>8s}')
    for cond in a.conds:
        for hd in HD_MODES:
            sub = [r for r in rows if r['condition'] == cond and r['hd_mode'] == hd]
            if not sub:
                continue
            m = lambda k: float(np.nanmean([r[k] for r in sub]))
            print(f'{cond+"/"+hd:<12s} {m("n_place"):>8.0f} {m("mean_fields"):>12.2f} '
                  f'{m("frac_multifield"):>9.2f} {m("sym_c2"):>8.2f} {m("sym_c4"):>8.2f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
