"""Mean place-field AREA as a function of prediction horizon k, C2 arena.

WHY THIS EXISTS. The manuscript reports field area stepping 30.6 -> 38.1 -> 38.5 -> 37.4 across
k in {0,1,3,5} (full encoding, C2 arena) to argue prediction sets field size as a SWITCH, not a
dial. An audit found no code producing those four numbers (AUDIT_PHASE_A.md, A2). This regenerates
them, using the SAME field-area method as field_stats.py (count_fields: smoothed map, half-peak
threshold, connected components >= 3 cells; place cell = Skaggs SI >= 0.3).

Checkpoints: horizon/k{0,1,3}/s2/<enc>/seed_XX (k=0,1,3) + hd_invariance/s2/<enc>/seed_XX (k=5).

MEMORY. One checkpoint at a time. hidden 20000 x 500 x 4B = 40 MB; maps 500 x 324 x 8B = 1 MB.
Peak < 200 MB. CPU-only.

    PYTHONPATH=. python3 analysis/field_area_horizon.py \
        --horizon-root <ckpts>/horizon --k5-root <ckpts>/hd_invariance \
        --data-root <data> --out Report/data/field_area_horizon.csv
"""
from __future__ import annotations
import argparse, csv, gc, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.analysis.run_spectrum import collect, model_from_checkpoint  # noqa: E402
from project5_symmetry.analysis.rate_maps import mean_rate_maps, spatial_information  # noqa: E402
from project5_symmetry.analysis.field_stats import count_fields  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def field_area_for_ckpt(p, ds, arena, n_states, si_thresh, dev):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    hd = ck['meta']['hd_mode']
    model = model_from_checkpoint(ck, dev)
    hidden, pos = collect(model, ds, hd, n_states, dev)
    maps, occ = mean_rate_maps(hidden, pos, arena)
    mask = occ == 0
    si = spatial_information(maps, occ)
    sizes = []
    for u in range(maps.shape[0]):
        if si[u] < si_thresh:
            continue
        nf, fs = count_fields(maps[u], mask)
        if nf >= 1:
            sizes.append(fs)
    del hidden, pos, maps, model
    gc.collect()
    return hd, len(sizes), (float(np.mean(sizes)) if sizes else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon-root', required=True)
    ap.add_argument('--k5-root', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--cond', default='s2')
    ap.add_argument('--encodings', nargs='+', default=['full', 'parity', 'axis', 'const'])
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--si-thresh', type=float, default=0.3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    ds = TrajectoryDataset(str(Path(a.data_root) / a.cond))

    jobs = []  # (k, ckpt_path)
    for k in (0, 1, 3):
        for enc in a.encodings:
            for p in sorted((Path(a.horizon_root) / f'k{k}' / a.cond / enc).glob('seed_*/ckpt_final.pt')):
                jobs.append((k, p))
    for enc in a.encodings:
        for p in sorted((Path(a.k5_root) / a.cond / enc).glob('seed_*/ckpt_final.pt')):
            jobs.append((5, p))

    rows = []
    for i, (k, p) in enumerate(jobs, 1):
        hd, n_place, area = field_area_for_ckpt(p, ds, a.arena, a.n_states, a.si_thresh, dev)
        rows.append({'k': k, 'hd_mode': hd, 'seed': int(p.parent.name.split('_')[1]),
                     'n_place': n_place, 'mean_field_area': round(area, 4)})
        print(f'  [{i}/{len(jobs)}] k={k} {hd}/seed{rows[-1]["seed"]:02d}  '
              f'n_place={n_place:3d}  field_area={area:.2f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"k":>3s} {"enc":<7s} {"field_area":>11s}')
    for k in (0, 1, 3, 5):
        for enc in a.encodings:
            sub = [r for r in rows if r['k'] == k and r['hd_mode'] == enc]
            if sub:
                print(f'{k:>3d} {enc:<7s} {np.nanmean([r["mean_field_area"] for r in sub]):>11.2f}')
    print(f'\nwrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
