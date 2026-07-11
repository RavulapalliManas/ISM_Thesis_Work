"""Replay sequenceness: is an offline trajectory an ordered sweep, or a random walk?

Coverage (run_replay.py) says how much ground an offline trajectory covers; it does not say
whether the visited cells are traversed in a spatially ordered sequence. This is the statistic
Levenstein-style replay claims turn on. For each offline rollout we decode a position path,
project it onto its own principal axis to get a 1-D coordinate x(t), and score sequenceness as
the rank correlation |Spearman(x(t), t)|: a coherent forward or reverse sweep scores high, a
random walk near zero.

Every score is tested against two nulls, both the standard ones for replay:

  time-shift   circularly shift x(t) against t. Destroys the temporal order while preserving
               the marginal distribution of decoded positions. The primary replay null.
  cell-shuffle permute unit identities before decoding, so the decoder reads a scrambled code.
               Destroys any learned population structure.

We report the observed sequenceness, the 95th percentile of the time-shift null, and the
fraction of rollouts whose score exceeds that null (the sequenceness rate). A network with no
ordered offline structure has a rate near 0.05.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_replay import offline_rollout, random_actions  # noqa: E402
from project5_symmetry.analysis.run_spectrum import (collect, identify,  # noqa: E402
                                                     model_from_checkpoint)
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else 0.0


def sequenceness(path, n_shift=200, seed=0):
    """|Spearman(x(t), t)| of the path's principal-axis coordinate, and the time-shift null.

    Returns (score, null95, exceeds) where `exceeds` is True if the score beats the 95th
    percentile of circularly time-shifted scores.
    """
    P = path - path.mean(0)
    # principal axis of the trajectory
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    x = P @ vt[0]
    t = np.arange(len(x))
    score = abs(_spearman(x, t))
    rng = np.random.default_rng(seed)
    null = []
    L = len(x)
    for _ in range(n_shift):
        sh = int(rng.integers(1, L))
        null.append(abs(_spearman(np.roll(x, sh), t)))
    null95 = float(np.percentile(null, 95))
    return score, null95, bool(score > null95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-states', type=int, default=15_000)
    ap.add_argument('--n-rollouts', type=int, default=40)
    ap.add_argument('--rollout-len', type=int, default=150)
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
        if cond not in ds:
            ds[cond] = TrajectoryDataset(str(Path(a.data_root) / cond))
        model = model_from_checkpoint(ck, dev)
        H, Pos = collect(model, ds[cond], hd, a.n_states, dev)
        sc = StandardScaler().fit(H)
        dec = Ridge(alpha=1.0).fit(sc.transform(H), Pos.astype(float))
        rng = np.random.default_rng(seed)
        # cell-shuffle null: feed the TRAINED decoder scrambled units at readout. Fitting and
        # predicting under the same permutation would be a no-op (least squares is invariant to
        # feature order), so the mismatch is essential.
        perm = rng.permutation(H.shape[1])

        obs_scores, exceeds, shuf_scores = [], [], []
        wake_scores = []
        for r in range(a.n_rollouts):
            h0 = torch.from_numpy(H[rng.integers(len(H))]).float()
            acts = random_actions(a.rollout_len, rng)
            Ht = offline_rollout(model, h0, acts).numpy()
            X = sc.transform(Ht)
            s_obs, null95, exc = sequenceness(dec.predict(X), seed=r)
            s_shuf, _, _ = sequenceness(dec.predict(X[:, perm]), seed=r)
            obs_scores.append(s_obs); exceeds.append(exc); shuf_scores.append(s_shuf)
        # wake reference: sequenceness of real trajectory segments
        for r in range(a.n_rollouts):
            j = rng.integers(len(Pos) - a.rollout_len)
            wake_scores.append(sequenceness(Pos[j:j + a.rollout_len].astype(float), seed=r)[0])

        rows.append({'condition': cond, 'hd_mode': hd, 'seed': seed,
                     'seq_offline': float(np.mean(obs_scores)),
                     'seq_rate': float(np.mean(exceeds)),
                     'seq_cellshuf': float(np.mean(shuf_scores)),
                     'seq_wake': float(np.mean(wake_scores))})
        r0 = rows[-1]
        print(f'  [{i}/{len(ckpts)}] {cond}/{hd}/seed{seed}  seq={r0["seq_offline"]:.3f} '
              f'rate={r0["seq_rate"]:.2f} cellshuf={r0["seq_cellshuf"]:.3f} '
              f'wake={r0["seq_wake"]:.3f}', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"cond/hd":<12s} {"seq":>7s} {"rate":>7s} {"cellshuf":>9s} {"wake":>7s}')
    for key in sorted({(r['condition'], r['hd_mode']) for r in rows}):
        sub = [r for r in rows if (r['condition'], r['hd_mode']) == key]
        m = lambda k: float(np.mean([r[k] for r in sub]))
        print(f'{key[0]+"/"+key[1]:<12s} {m("seq_offline"):>7.3f} {m("seq_rate"):>7.2f} '
              f'{m("seq_cellshuf"):>9.3f} {m("seq_wake"):>7.3f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
