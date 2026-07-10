"""Offline dynamics: does the network generate coherent spatial trajectories with no input?

\\citet{levenstein2024} report that offline simulations emerge only in networks that use
recurrent connections and head-direction information to predict future observations.

This module was written to look for a DISSOCIATION: symmetry is resolved at k = 1, so if
replay needed a longer horizon the two would come apart on Levenstein's own axis. It does
not. Measured (s2, n = 6 per cell, coverage as a fraction of a wake path of equal length):

    k = 0 (autoencoder)   all four HD encodings      0.30 - 0.37 x wake
    k = 1   full 1.30   parity 0.80   axis 0.45   const 0.31
    k = 3   full 1.11   parity 0.80   axis 0.49   const 0.30
    k = 5   full 1.04   parity 0.82   axis 0.50   const 0.25

Replay appears at exactly the horizon where the fold is resolved (k = 1) and does not
improve with a longer one, and it fails under the same HD ablations that make the code
fold. Both are step functions in k, and they step together. That is a stronger agreement
with Levenstein than the dissociation this file went looking for.

How offline rollout works here
------------------------------
The network is driven by its OWN predicted observation instead of the environment's:

    o_hat_t = sigmoid(W_out h_t)
    h_{t+1} = relu( layernorm( W_in [o_hat_t ; a_t] + W h_t ) + bias )

with `a_t` a random-walk action sequence (the "virtual head direction signal"). This is the
same recurrence the model runs online; `manual_rollout` reproduces the model's own hidden
states bit-for-bit when fed the real observations, which is the check that the loop is right.

Readouts
--------
    step        median |Delta position| per offline step, from a ridge decoder fit on wake data
    coverage    fraction of arena cells the offline trajectory visits
    continuity  fraction of steps with |Delta position| <= 2 cells

Every readout is reported against the SAME number of wake steps (`wake_*`) and against a
shuffle of the wake positions (`shuf_*`). A network with no offline structure lands on the
shuffle. Coverage in particular is meaningless alone: a 200-step wake path cannot visit more
than 200 of the 324 cells either, so `off_cov` must be read as a fraction of `wake_cov`, and
`shuf_cov` is the spread-out ceiling. Continuity runs the other way -- wake is the ceiling,
shuffle the floor.

For a FOLDED network the raw decoded path should look like it teleports between orbit-mates,
while the same path decoded in the fundamental domain should be continuous -- replay on the
quotient. Both are reported.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_phase_decoding import canonical  # noqa: E402
from project5_symmetry.analysis.run_spectrum import (collect, identify,  # noqa: E402
                                                     model_from_checkpoint)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def _cell_params(model):
    c = model.rnn.cell
    return (c.weight_ih.detach(), c.weight_hh.detach(), c.bias.detach(),
            model.W_out.detach())


def _layernorm(x):
    """The project's hand-rolled norm: (x - mean) / (std + 1e-4). NOT F.layer_norm."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-4)


@torch.no_grad()
def manual_rollout(model, obs, act, h0):
    """Reproduce the model's own recurrence, driven by the REAL observations.

    Exists to prove the offline loop below is the same dynamical system.
    """
    W_ih, W_hh, bias, _ = _cell_params(model)
    h, out = h0, []
    for t in range(act.shape[0]):
        x = torch.cat([obs[t], act[t]], dim=-1)
        h = torch.relu(_layernorm(x @ W_ih.t() + h @ W_hh.t()) + bias)
        out.append(h)
    return torch.stack(out)


@torch.no_grad()
def offline_rollout(model, h0, acts):
    """Drive the network with its own prediction. No environment."""
    W_ih, W_hh, bias, W_out = _cell_params(model)
    h, out = h0, []
    for t in range(acts.shape[0]):
        o_hat = torch.sigmoid(h @ W_out.t())
        x = torch.cat([o_hat, acts[t]], dim=-1)
        h = torch.relu(_layernorm(x @ W_ih.t() + h @ W_hh.t()) + bias)
        out.append(h)
    return torch.stack(out)


def random_actions(n, rng, speed_p=0.7):
    """SpeedHD: [speed, onehot(heading)] with a persistent heading, as in the wake policy."""
    a = np.zeros((n, 5), dtype=np.float32)
    h = rng.integers(4)
    for t in range(n):
        if rng.random() < 0.25:
            h = (h + rng.choice([-1, 1])) % 4
        a[t, 0] = 1.0 if rng.random() < speed_p else 0.0
        a[t, 1 + h] = 1.0
    return torch.from_numpy(a)


def path_stats(P, arena=18, thresh=2.0):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cells = {(int(round(x)), int(round(y))) for x, y in P}
    return {'step': float(np.median(d)),
            'continuity': float((d <= thresh).mean()),
            'coverage': len(cells) / (arena * arena)}


READOUTS = {'step': 'step', 'continuity': 'cont', 'coverage': 'cov'}


def summary_row(meta, off_raw, off_dom, wake, shuffle):
    """One row per model: every readout carries its wake and shuffle baseline.

    Built by looping over READOUTS rather than typed out per metric, because the hand-written
    version silently omitted `wake_cov` -- leaving `off_cov = 0.358` with nothing to compare
    it to. A 200-step offline path cannot cover more arena than a 200-step wake path, so a
    coverage number without `wake_cov` next to it is not a result.
    """
    mean = lambda lst, key: float(np.mean([d[key] for d in lst]))
    row = dict(meta)
    for key, short in READOUTS.items():
        row[f'off_{short}'] = mean(off_raw, key)
        row[f'dom_{short}'] = mean(off_dom, key)
        row[f'wake_{short}'] = wake[key]
        row[f'shuf_{short}'] = shuffle[key]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-states', type=int, default=20_000)
    ap.add_argument('--n-rollouts', type=int, default=20)
    ap.add_argument('--rollout-len', type=int, default=200)
    ap.add_argument('--arena', type=int, default=18)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ckpts = sorted(Path(a.runs).rglob('ckpt_final.pt'))
    if not ckpts:
        raise SystemExit(f'no ckpt_final.pt under {a.runs}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    ds, rows = {}, []
    for i, p in enumerate(ckpts, 1):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cond, hd, seed = identify(p, ck['meta'])
        k = ck['meta']['k']
        if cond not in ds:
            ds[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        H, P = collect(model, ds[cond], hd, a.n_states, dev)

        sc = StandardScaler().fit(H)
        dec_raw = Ridge(alpha=1.0).fit(sc.transform(H), P.astype(float))
        dec_dom = Ridge(alpha=1.0).fit(sc.transform(H), canonical(P, 'c2').astype(float))

        rng = np.random.default_rng(seed)
        off_raw, off_dom = [], []
        for _ in range(a.n_rollouts):
            h0 = torch.from_numpy(H[rng.integers(len(H))]).float()
            acts = random_actions(a.rollout_len, rng)
            Ht = offline_rollout(model, h0, acts).numpy()
            X = sc.transform(Ht)
            off_raw.append(path_stats(dec_raw.predict(X), a.arena))
            off_dom.append(path_stats(dec_dom.predict(X), a.arena))

        # wake reference and a shuffle floor, on the same decoder
        wake = path_stats(P[:a.rollout_len].astype(float), a.arena)
        sh = P[rng.permutation(len(P))[:a.rollout_len]].astype(float)
        shuffle = path_stats(sh, a.arena)

        meta = {'condition': cond, 'hd_mode': hd, 'k': k, 'seed': seed}
        rows.append(summary_row(meta, off_raw, off_dom, wake, shuffle))
        r = rows[-1]
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/k{k}/seed{seed}  '
              f'offline step={r["off_step"]:.2f} cont={r["off_cont"]:.3f} | '
              f'domain cont={r["dom_cont"]:.3f} | wake {r["wake_cont"]:.3f} '
              f'shuffle {r["shuf_cont"]:.3f} | '
              f'cov {r["off_cov"]:.3f} (wake {r["wake_cov"]:.3f}, '
              f'shuf {r["shuf_cov"]:.3f})', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
