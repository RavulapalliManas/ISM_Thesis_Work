#!/usr/bin/env python3
"""Checkpoint -> C4 isotypic spectrum, one row per trained model.

For each checkpoint: roll the model over trajectories from its own arena, collect
hidden states and positions, build per-unit rate maps, and decompose each map into
the four C4 characters. Reports, per model:

    P0..P3   normalised isotypic power fractions (sum to 1)
    RA       = P0 - P2      (the Proposition: rotational autocorrelation)
    odd      = P1 + P3      power in the characters that a 180-deg rotation flips

`odd` is the quantity the HD-invariance experiment turns on. A code that has folded
onto the C2 quotient puts ZERO power in the odd characters: f(x) == f(R^2 x) kills
exactly k=1 and k=3. RA cannot see this -- it reads ~0 for a C2-folded code and ~0
for noise alike (see test_isotypic.py::test_ra_is_blind_to_c2_structure).

The HD encoding is re-applied at evaluation time from the checkpoint's own metadata:
a model trained on axis-collapsed headings must be *evaluated* on them, or its inputs
are off-distribution and the spectrum is meaningless. Checkpoints predating the HD
experiment carry no `hd_mode` and are read as `full`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.isotypic import (isotypic_spectrum, odd_power,  # noqa: E402
                                                 ra_from_spectrum)
from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402
from utils.Architectures import load_prnn_state_dict, pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager  # noqa: E402


CONDITIONS = ('s1', 's2', 's4')
HD_MODES = ('full', 'axis', 'parity', 'const')


def identify(path: Path, meta: dict) -> tuple[str, str, int]:
    """(condition, hd_mode, seed), from metadata where present, else from the path.

    Group A's checkpoints predate the HD experiment and record none of these in
    `meta` -- only the directory layout .../{condition}/seed_NN/ knows. Later runs
    put all three in `meta`. Prefer meta; fall back to the path; never guess.
    """
    parts = path.parts
    cond = meta.get('condition') or next((p for p in parts if p in CONDITIONS), None)
    hd = meta.get('hd_mode') or next((p for p in parts if p in HD_MODES), 'full')
    seed = meta.get('seed')
    if seed is None:
        s = next((p for p in parts if p.startswith('seed_')), None)
        seed = int(s.split('_')[1]) if s else -1
    if cond is None:
        raise ValueError(f'cannot identify condition for {path}')
    return cond, hd, int(seed)


def model_from_checkpoint(ckpt, device):
    """Rebuild the trained architecture from its metadata.

    Anything that changes the shape or wiring of the network is REQUIRED -- a wrong
    guess would load weights into the wrong model and produce a plausible spectrum.
    `pred_offset` is the one field Group A's checkpoints omit; every driver in this
    repo sets it to 0, and it does not affect the eval-time forward, so it defaults.
    """
    m = ckpt['meta']
    missing = [f for f in ('obs_size', 'k', 'hidden_size', 'trunc') if f not in m]
    if missing:
        raise KeyError(f'checkpoint metadata is missing {missing}; refusing to guess')
    model = pRNN_th(obs_size=m['obs_size'], act_size=5, k=m['k'],
                    hidden_size=m['hidden_size'], cell=LayerNormRNNCellEager,
                    dropp=m.get('dropout_p', 0.0), trunc=m['trunc'], neuralTimescale=2,
                    predOffset=m.get('pred_offset', 0),
                    hidden_init_sigma=m.get('hidden_init_sigma', 0.1))
    load_prnn_state_dict(model, ckpt['model'])
    return model.to(device).eval()


@torch.no_grad()
def collect(model, dataset, hd_mode, n_states, device, seed=0):
    """Hidden states and positions, with the model's own HD encoding applied."""
    hs, ps, total = [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, _ = dataset[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        take = min(h.shape[0], n_states - total)
        hs.append(h[:take])
        ps.append(pos.numpy()[:h.shape[0]][:take])
        total += take
        if total >= n_states:
            break
    hidden, positions = np.concatenate(hs, 0), np.concatenate(ps, 0)
    # The dataset stores positions as float32; rate_maps indexes with them. They are
    # 1-based integer grid coordinates, so the cast is exact -- assert it rather than
    # silently rounding a continuous position onto the wrong bin.
    assert np.all(positions == np.floor(positions)), 'positions are not integral'
    return hidden, positions.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='root holding */ckpt_final.pt')
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--ckpt-name', default='ckpt_final.pt')
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    ckpts = sorted(Path(a.runs).rglob(a.ckpt_name))
    if not ckpts:
        raise SystemExit(f'no {a.ckpt_name} under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    datasets = {}
    rows = []
    for i, path in enumerate(ckpts, 1):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        meta = ck['meta']
        cond, hd, seed = identify(path, meta)
        if cond not in datasets:
            datasets[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, device)
        hidden, pos = collect(model, datasets[cond], hd, a.n_states, device)
        P = isotypic_spectrum(hidden, pos, arena=a.arena)
        rows.append({'condition': cond, 'hd_mode': hd, 'k': meta['k'],
                     'seed': seed, 'step': ck['step'],
                     'P0': P[0], 'P1': P[1], 'P2': P[2], 'P3': P[3],
                     'RA': ra_from_spectrum(P), 'odd': odd_power(P),
                     'path': str(path)})
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/k{meta["k"]}/seed{seed:02d}  '
              f'RA={rows[-1]["RA"]:+.4f}  odd={rows[-1]["odd"]:.4f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
