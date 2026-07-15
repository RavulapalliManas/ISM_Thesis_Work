"""Room decodability conditioned on steps-since-room-entry, TWO-room maze, BOTH arrangements.

WHY THIS EXISTS. The Methods report a corridor-memory confound control: room decoding conditioned
on steps since the animal entered its current room, translation decaying 0.63 -> 0.52 and rotation
rising 0.84 -> 1.00. An audit (AUDIT_PHASE_A.md, A2) found the existing dwell script
(run_compartments_horizon.py) is TRANSLATION-ONLY (line 121) and its numbers don't match, and no
code produces the rotation series. This regenerates BOTH, reusing that script's own
steps_since_entry / dwell_conditioned_metrics verbatim.

The compartment trajectory data is not on the backup drive (only checkpoints), so it is regenerated
here from the identical env factory (make_compartment_env, F=7, T=200, seed=0) used to train the
networks. Fresh random-walk trajectories through the SAME maze give statistically equivalent (not
bit-identical) decode statistics -- adequate for a confound control whose claim is a qualitative
pattern (translation decays toward chance, rotation rises to ceiling).

MEMORY. Dataset cache ~200 MB/mode (n_traj x 200 x 147 floats); hidden 40000 x 500 x 4B = 80 MB.
One checkpoint at a time. CPU-only.

    PYTHONPATH=. python3 analysis/corridor_dwell.py --ckpt-root <ckpts>/multi/compartment \
        --data-dir <writable>/compartment_data --out Report/data/corridor_dwell.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.environments.compartment_arenas import make_compartment_env  # noqa: E402
from project5_symmetry.analysis.run_compartments import label_rooms  # noqa: E402
from project5_symmetry.analysis.run_compartments_horizon import (collect_with_bounds,  # noqa: E402
                                                                 dwell_conditioned_metrics,
                                                                 DWELL_BINS)
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

F, T = 7, 200


def ensure_mode_data(mode, data_dir, n_traj, workers):
    out = Path(data_dir) / mode
    have = len(list(out.glob('traj_*.npz'))) if out.exists() else 0
    if have < n_traj:
        print(f'  generating {n_traj} {mode} trajectories -> {out} (have {have})', flush=True)
        generate_dataset(make_compartment_env(mode, F=F, seed=0), n_traj=n_traj, T=T,
                         out_dir=str(out), n_workers=workers, desc=mode,
                         env_factory=make_compartment_env,
                         factory_kwargs={'mode': mode, 'F': F, 'seed': 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True, help='.../multi/compartment')
    ap.add_argument('--data-dir', required=True, help='writable dir for regenerated trajectories')
    ap.add_argument('--modes', nargs='+', default=['translation', 'rotation'])
    ap.add_argument('--n-traj', type=int, default=1000)
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--gen-workers', type=int, default=6)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows = []
    for mode in a.modes:
        data = ensure_mode_data(mode, a.data_dir, a.n_traj, a.gen_workers)
        ds = TrajectoryDataset(str(data))
        ckpts = sorted((Path(a.ckpt_root) / mode).glob('seed_*/ckpt_final.pt'))
        print(f'{mode}: {len(ckpts)} checkpoints', flush=True)
        for i, p in enumerate(ckpts, 1):
            ck = torch.load(p, map_location='cpu', weights_only=False)
            seed = ck['meta']['seed']
            model = model_from_checkpoint(ck, torch.device('cpu'))
            H, P, bounds = collect_with_bounds(model, ds, a.n_states, seed)
            room, loc = label_rooms(P, mode)
            metrics = dwell_conditioned_metrics(H, room, loc, bounds, seed=seed)
            for dwell_bin, m in metrics.items():
                rows.append({'mode': mode, 'seed': seed, 'dwell_bin': dwell_bin, **m})
            steady = metrics[f'{DWELL_BINS[-1][0]}-inf']
            entry = metrics[f'{DWELL_BINS[0][0]}-{DWELL_BINS[0][1]}']
            print(f'  [{i}/{len(ckpts)}] {mode}/seed{seed}  entry(0-2) gen={entry["room_gen"]:.3f}'
                  f'  steady(>10) gen={steady["room_gen"]:.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"mode":<12s} {"dwell":>8s} {"room_gen":>9s} {"room_seen":>10s}')
    for mode in a.modes:
        for lo, hi in DWELL_BINS:
            lab = f'{lo}-{hi if hi is not None else "inf"}'
            sub = [r for r in rows if r['mode'] == mode and r['dwell_bin'] == lab]
            if sub:
                g = np.nanmean([r['room_gen'] for r in sub])
                s = np.nanmean([r['room_seen'] for r in sub])
                print(f'{mode:<12s} {lab:>8s} {g:>9.3f} {s:>10.3f}')
    print(f'\nwrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
