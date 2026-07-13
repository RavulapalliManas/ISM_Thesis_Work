"""Referee item 1: the graded-cue SNR sweep -- analysis.

Trained by experiments/run_multi.py --groups weakbreak (see WEAKBREAK_* constants there for
the design: fixed pixel noise sigma=0.05, room-B tint eps swept over 8 values spanning
single-step d' = eps*sqrt(N)/sigma in {0 .. 3}, both applied in float, downstream of 8-bit
quantisation, so the analytic Bayes bound acc_max(d', T) = Phi(0.5 * d' * sqrt(T)) is a smooth
sigmoid rather than the step function the old amplitude-only cue produced).

Per checkpoint, four readouts (reusing the same primitives as run_compartments.py, which is
literally the same measurement on a different room-B condition):

  room_gen, room_seen   which compartment (decode_room) -- the position-unfolds readout
  within_r2             ridge R^2 for position WITHIN a room -- alive-code control
  repetition            median corr(rate map A, rate map B) over matched local cells --
                         the SAME thing run_compartments.py calls "orbit-PV correlation" /
                         Leutgeb-style remapping metric for this maze; not a separate readout
  rate_asymmetry        NEW. median over the top-100-variance units of
                         |rate_A - rate_B| / (rate_A + rate_B), i.e. relative firing-rate
                         asymmetry between rooms, with NO position matching (unlike
                         `repetition`, which requires shared local cells). Predicted (Spiers'
                         signature): rate_asymmetry departs from 0 at LOWER d'/sigma than
                         room_gen departs from chance, while `repetition` stays high throughout
                         -- rate discrimination appearing before position does.

The predicted curve overlay (acc_max at T=1, single observation, and T=10, an representative
dwell) is printed for reference; it is a bound on a DIFFERENT quantity (single-image Bayes-
optimal room discriminability) than room_gen (a trained decoder's cross-validated accuracy
using the recurrent state's ~10-step room dwell), so treat it as a sanity ceiling, not an
equality target -- room_gen exceeding the T=1 bound is expected and fine; it should stay under
the T=10-ish bound.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_compartments import (collect, decode_room,  # noqa: E402
                                                          label_rooms, repetition_index)
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

SIGMA = 0.05
N_PIXELS = 7 * 7 * 3


def rate_asymmetry(H, room, top_n=100):
    """median_u |rate_A(u) - rate_B(u)| / (rate_A(u) + rate_B(u)), over the top-variance units.
    No position matching -- unlike `repetition_index`, every room-interior state counts."""
    a, b = H[room == 0], H[room == 1]
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    rate_a, rate_b = a.mean(0), b.mean(0)
    var = H.var(0)
    top = np.argsort(var)[-top_n:]
    denom = np.abs(rate_a[top]) + np.abs(rate_b[top])
    ok = denom > 1e-8
    if ok.sum() == 0:
        return float('nan')
    return float(np.median(np.abs(rate_a[top][ok] - rate_b[top][ok]) / denom[ok]))


def acc_max(d_prime, T):
    return float(norm.cdf(0.5 * d_prime * np.sqrt(T)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data')
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
        if meta.get('group') != 'weakbreak':
            continue
        d_prime, tint, seed = meta['d_prime'], meta['tint'], meta['seed']
        store = meta['store']                                   # e.g. 'weakbreak/d0p43'
        if store not in ds_cache:
            ds_cache[store] = TrajectoryDataset(str(Path(a.data_root) / store))
        model = model_from_checkpoint(ck, torch.device('cpu'))

        H, P = collect(model, ds_cache[store], a.n_states, seed=seed)
        room, loc = label_rooms(P, 'translation')
        gen, seenv, within = decode_room(H, room, loc, seed=seed)
        rep, ncells = repetition_index(H, room, loc)
        asym = rate_asymmetry(H, room)

        rows.append({'d_prime': d_prime, 'tint': tint, 'seed': seed, 'room_gen': gen,
                     'room_seen': seenv, 'within_r2': within, 'repetition': rep,
                     'rate_asymmetry': asym, 'n_cells_matched': ncells,
                     'n_states_in_rooms': int((room >= 0).sum()),
                     'acc_max_T1': acc_max(d_prime, 1), 'acc_max_T10': acc_max(d_prime, 10)})
        print(f"  [{i}/{len(ckpts)}] d'={d_prime:.2f} seed={seed:02d}  room_gen={gen:.4f}  "
              f'repetition={rep:+.4f}  rate_asym={asym:.4f}  '
              f'(acc_max T1={acc_max(d_prime,1):.3f} T10={acc_max(d_prime,10):.3f})', flush=True)

    if not rows:
        raise SystemExit(f"no weakbreak checkpoints found under {a.runs}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"d_prime":>8s} {"room_gen":>9s} {"room_seen":>10s} {"repetition":>11s} '
          f'{"rate_asym":>10s} {"acc_max_T1":>11s}')
    for d in sorted({r['d_prime'] for r in rows}):
        sub = [r for r in rows if r['d_prime'] == d]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f'{d:>8.2f} {m("room_gen"):>9.4f} {m("room_seen"):>10.4f} '
              f'{m("repetition"):>11.4f} {m("rate_asymmetry"):>10.4f} {m("acc_max_T1"):>11.4f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
