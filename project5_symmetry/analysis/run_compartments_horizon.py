"""Referee item 2: does room decodability in the two-room maze depend on the prediction
horizon k?

The corridor connecting the two compartments is not length-matched between arrangements
(Methods): translation's link is 27 cells, rotation's is 57. A longer, more distinctive
corridor is a channel by which recurrent memory of the just-completed transit -- not the
compass -- could carry room identity into the following room-interior states. The existing
mitigation conditions on steps-since-room-entry and reports the >10-step steady state
(translation: 0.63 at entry decaying to 0.52 by ~10 steps). That was checked at a single k=5.

This script repeats the same entry-conditioned measurement across k in {0,1,3,5,10}, each
trained separately (train_compartment_horizon.py -- k is a distinct graph, so a distinct job
per k). Two readings are both informative:

  room decodability RISES with k   -> the residual is horizon-limited: the corridor cue is
                                       present but only recruited by an objective that predicts
                                       far enough ahead. A second instance of "information
                                       present != information recruited" (Limitations).
  room decodability is FLAT in k   -> the cue is too weak to recruit at any tested horizon;
                                       the fold is architecture-robust to horizon choice.

Either way, always report the >10-step steady-state value as primary (matching the existing
Methods convention), since early bins are the ones any corridor-memory confound would inflate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_compartments import (decode_room,  # noqa: E402
                                                          label_rooms, repetition_index)
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

# (bin_lo, bin_hi_exclusive_or_None); the last bin is the >10-step steady state, primary per
# the existing Methods convention.
DWELL_BINS = ((0, 2), (2, 5), (5, 10), (10, None))


def steps_since_entry(room: np.ndarray) -> np.ndarray:
    """Steps since `room` last changed value, within THIS sequence only. -1 (non-room,
    corridor) states get -1; the state where room identity actually changes gets 0."""
    since = np.full(len(room), -1, dtype=int)
    prev, count = None, -1
    for i, r in enumerate(room):
        if r < 0:
            prev, count = None, -1
        elif r != prev:
            prev, count = r, 0
        else:
            count += 1
        since[i] = count
    return since


@torch.no_grad()
def collect_with_bounds(model, ds, n_states, seed=0):
    """Like run_compartments.collect, but also returns each source trajectory's (start, end)
    slice into the concatenated arrays, so steps_since_entry is never computed across a
    trajectory boundary."""
    hs, ps, bounds, tot = [], [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(ds), generator=g):
        obs, act, pos, _ = ds[int(idx)]
        _, h, _ = model(obs.unsqueeze(0), act.unsqueeze(0))
        h = h.squeeze(0).numpy()
        take = min(h.shape[0], n_states - tot)
        hs.append(h[:take]); ps.append(pos.numpy()[:h.shape[0]][:take])
        bounds.append((tot, tot + take))
        tot += take
        if tot >= n_states:
            break
    return np.concatenate(hs, 0), np.concatenate(ps, 0).astype(int), bounds


def dwell_conditioned_metrics(H, room, loc, bounds, seed=0):
    """decode_room + repetition_index, each restricted to states in a given dwell bin."""
    since = np.concatenate([steps_since_entry(room[a:b]) for a, b in bounds])
    out = {}
    for lo, hi in DWELL_BINS:
        m = (since >= lo) & (since < hi if hi is not None else True)
        label = f'{lo}-{hi if hi is not None else "inf"}'
        if m.sum() < 50:
            out[label] = {'room_gen': float('nan'), 'room_seen': float('nan'),
                          'within_r2': float('nan'), 'repetition': float('nan'), 'n': int(m.sum())}
            continue
        gen, seenv, within = decode_room(H[m], room[m], loc[m], seed=seed)
        rep, ncells = repetition_index(H[m], room[m], loc[m])
        out[label] = {'room_gen': gen, 'room_seen': seenv, 'within_r2': within,
                      'repetition': rep, 'n': int(m.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='root containing k*/translation/seed_XX/ckpt_final.pt')
    ap.add_argument('--data-root', default='/root/data/compartment')
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds_cache, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        meta = ck['meta']
        mode, seed, k = meta['mode'], meta['seed'], meta['k']
        if mode != 'translation':
            continue                                    # this sweep is translation-only (spec)
        if mode not in ds_cache:
            ds_cache[mode] = TrajectoryDataset(str(Path(a.data_root) / mode))
        model = model_from_checkpoint(ck, torch.device('cpu'))
        H, P, bounds = collect_with_bounds(model, ds_cache[mode], a.n_states, seed)
        room, loc = label_rooms(P, mode)
        metrics = dwell_conditioned_metrics(H, room, loc, bounds, seed=seed)
        for dwell_bin, m in metrics.items():
            rows.append({'k': k, 'seed': seed, 'dwell_bin': dwell_bin, **m})
        steady = metrics[f'{DWELL_BINS[-1][0]}-inf']
        print(f'  [{i}/{len(ckpts)}] k={k} seed={seed}  steady(>10) room_gen={steady["room_gen"]:.4f} '
              f'room_seen={steady["room_seen"]:.4f}  repetition={steady["repetition"]:+.4f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"k":>4s} {"dwell_bin":>10s} {"room_gen":>9s} {"room_seen":>10s} {"repetition":>11s}')
    for k in sorted({r['k'] for r in rows}):
        for lo, hi in DWELL_BINS:
            label = f'{lo}-{hi if hi is not None else "inf"}'
            sub = [r for r in rows if r['k'] == k and r['dwell_bin'] == label]
            mean = lambda key: float(np.nanmean([r[key] for r in sub]))
            print(f'{k:>4d} {label:>10s} {mean("room_gen"):>9.4f} {mean("room_seen"):>10.4f} '
                  f'{mean("repetition"):>11.4f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
