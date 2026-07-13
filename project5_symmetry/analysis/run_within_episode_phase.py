"""Experiment 0: within-episode symmetry breaking.

Every existing orbit-phase number (e.g. the learned-compass C2 result, phase_acc = 0.526,
chance 0.500) is measured ACROSS episodes: states are pooled from ~10,000 independent
episodes, each starting at a random position and heading, then a decoder is asked to recover
the ABSOLUTE orbit phase (which of {x, R^2 x} produced this state) from the hidden state,
generalising across held-out orbits. Chance there does not distinguish two things:

  (b) the network folds the code INSTANTANEOUSLY -- at every timestep, in every episode, the
      hidden state genuinely satisfies h(x) == h(R^2 x), so no decoder anywhere could do
      better than chance, at any timescale.
  (a) the network maintains a COHERENT INTERNAL FRAME within one episode (a latched "which
      copy am I in, relative to where I started" bit, carried by the recurrent state) that
      only looks like chance in the pooled, across-episode measurement because each episode
      starts at an independently random heading/position, randomising the mapping between the
      internal frame and the true world frame. This does not violate the Theorem: the bound
      is on a decoder that is a fixed function of the INPUT sequence, not on a quantity carried
      in the recurrent state across time. The existing domain_r2 = 0.89 for the learned
      compass in C2 (position WITHIN the fundamental domain is well decoded) is suggestive of a
      coherent internal frame; this script tests it directly.

The test: relabel orbit phase RELATIVE to each episode's own state at t=0:

    z(t) = phase_abs(t) XOR phase_abs(0)          -- "same copy as I started in" (0) vs
                                                       "flipped copy" (1)

z(t) is well-defined per-episode regardless of the network's absolute frame. If z(t) is
decodable from h(t) by a classifier trained on OTHER episodes (GroupKFold by episode, so the
decoder never sees the episode it is tested on -- it cannot memorise a per-episode offset),
that is direct evidence of outcome (a). If z(t) stays at chance, that is outcome (b). Binning
decode accuracy by steps-since-episode-start distinguishes (a)/(b) from:

  (c) decodable but drifting -- accuracy high near t=0, decaying toward chance later in the
      episode. The decay rate is the frame's half-life.

Run for whichever encodings are asked for (learned / axis / const); the theoretical prediction
is asymmetric between them (see Report/): a path-integrated compass has a mechanism (its own
sustained integration) that could carry a persistent relative frame, whereas axis/const receive
a fresh (folded) function of the TRUE, world-anchored heading every step and have no analogous
reason to need one -- but this is an empirical question for each, not assumed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.analysis.run_phase_decoding import (ARENA, decode,  # noqa: E402
                                                           orbit_and_phase)
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa: E402
from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa: E402

# Steps-since-episode-start bins for the drift check.
DRIFT_BINS = ((0, 10), (10, 25), (25, 50), (50, 100), (100, None))


@torch.no_grad()
def collect_with_bounds(model, dataset, hd_mode, n_states, device, seed=0):
    """Like run_spectrum.collect, but also returns each episode's (start, end) slice into
    the concatenated arrays, so z(t) is never computed across an episode boundary."""
    hs, ps, bounds, total = [], [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, _ = dataset[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        take = min(h.shape[0], n_states - total)
        hs.append(h[:take])
        ps.append(pos.numpy()[:h.shape[0]][:take])
        bounds.append((total, total + take))
        total += take
        if total >= n_states:
            break
    hidden = np.concatenate(hs, 0)
    positions = np.concatenate(ps, 0)
    assert np.all(positions == np.floor(positions)), 'positions are not integral'
    return hidden, positions.astype(np.int64), bounds


def relabel_within_episode(pos, bounds, group='c2', arena=ARENA):
    """z(t) = phase_abs(t) XOR phase_abs(0) within each episode; episode_id and
    steps-since-start per state, for GroupKFold and the drift breakdown."""
    _, phase_abs = orbit_and_phase(pos, group, arena)
    z = np.empty(len(pos), dtype=np.int64)
    episode_id = np.empty(len(pos), dtype=np.int64)
    t_in_episode = np.empty(len(pos), dtype=np.int64)
    for e, (a, b) in enumerate(bounds):
        z[a:b] = phase_abs[a:b] ^ phase_abs[a]
        episode_id[a:b] = e
        t_in_episode[a:b] = np.arange(b - a)
    return z, episode_id, t_in_episode


def _balanced_acc(y_true, y_pred):
    """0.5*(TPR+TNR). Chance is 0.5 regardless of class prevalence -- unlike raw accuracy,
    which a majority-class-only classifier can inflate for free whenever P(z=1) != 0.5."""
    out = []
    for c in (0, 1):
        m = y_true == c
        if m.sum() == 0:
            continue
        out.append((y_pred[m] == c).mean())
    return float(np.mean(out)) if out else float('nan')


def decode_within_episode(H, z, episode_id, t_in_episode, n_splits=5, seed=0):
    """logistic decoder for z(t), GroupKFold by episode -- the decoder never sees a test
    episode's own states during training, so it cannot memorise a per-episode offset.

    z is NOT balanced by construction: P(z=1) rises from ~0.08 at t=0-10 to ~0.46 by t=100+,
    purely from position continuity under a random walk (an agent that hasn't moved far is
    still very likely on the side of the arena it started on), independent of anything the
    hidden state encodes. A plain-accuracy decoder can therefore score ~0.92 at t=0-10 by
    ALWAYS predicting z=0 and using zero information from H -- this is exactly what an early
    version of this function did, and the "result" was indistinguishable from the trivial
    majority-class baseline at every bin. Two fixes, both required: (1) class_weight='balanced'
    so the fitted decision boundary is not itself dominated by the skewed prior, (2) report
    BALANCED accuracy (chance = 0.5 at any prevalence), never raw accuracy, as the readout.
    The majority-class baseline is returned alongside every number specifically so this
    confound cannot silently reappear if the bins or the policy change later.
    """
    n_groups = min(n_splits, len(np.unique(episode_id)))
    preds = np.full(len(z), -1, dtype=np.int64)
    for tr, te in GroupKFold(n_splits=n_groups).split(H, z, groups=episode_id):
        sc = StandardScaler().fit(H[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(
            sc.transform(H[tr]), z[tr])
        preds[te] = clf.predict(sc.transform(H[te]))
    assert (preds >= 0).all(), 'every state should fall in exactly one held-out fold'

    overall = _balanced_acc(z, preds)
    by_bin = {}
    for lo, hi in DRIFT_BINS:
        m = (t_in_episode >= lo) & (t_in_episode < hi if hi is not None else True)
        label = f'{lo}-{hi if hi is not None else "inf"}'
        if m.sum() < 50:
            by_bin[label] = {'balanced_acc': float('nan'), 'majority_baseline': float('nan'),
                             'p_z1': float('nan'), 'n': int(m.sum())}
            continue
        p1 = float(z[m].mean())
        by_bin[label] = {'balanced_acc': _balanced_acc(z[m], preds[m]),
                         'majority_baseline': max(p1, 1 - p1), 'p_z1': p1, 'n': int(m.sum())}
    return overall, by_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--condition', default='s2', help='arena condition, e.g. s2 for C2')
    ap.add_argument('--hd-modes', nargs='+', default=['learned', 'axis', 'const'])
    ap.add_argument('--ckpt-name', default='ckpt_final.pt')
    ap.add_argument('--n-states', type=int, default=40_000)
    ap.add_argument('--group', default='c2', choices=['c2', 'c4'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device('cpu')
    dataset = TrajectoryDataset(str(Path(a.data_root) / a.condition))

    ckpts = sorted(Path(a.runs).rglob(a.ckpt_name))
    if not ckpts:
        raise SystemExit(f'no {a.ckpt_name} under {a.runs}')

    rows = []
    for i, path in enumerate(ckpts, 1):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        hd_mode = ck['meta'].get('hd_mode', ck['meta'].get('hd'))
        if hd_mode not in a.hd_modes:
            continue
        seed = ck['meta']['seed']
        arena = int(ck['meta'].get('arena_size', ARENA))
        model = model_from_checkpoint(ck, device)

        hidden, pos, bounds = collect_with_bounds(model, dataset, hd_mode, a.n_states, device,
                                                  seed=seed)
        # Existing across-episode absolute-phase measure, on the SAME hidden states, for a
        # direct apples-to-apples comparison rather than a re-quoted Table 2 number.
        across_acc, raw_r2, dom_r2 = decode(hidden, pos, group=a.group, arena=arena, seed=seed)

        z, episode_id, t_in_ep = relabel_within_episode(pos, bounds, group=a.group, arena=arena)
        within_acc, by_bin = decode_within_episode(hidden, z, episode_id, t_in_ep, seed=seed)

        row = {'hd_mode': hd_mode, 'seed': seed, 'condition': a.condition, 'group': a.group,
               'chance': 0.5, 'across_episode_phase_acc': across_acc,
               'within_episode_balanced_acc': within_acc, 'raw_r2': raw_r2, 'domain_r2': dom_r2,
               'n_episodes': len(bounds), 'n_states': hidden.shape[0]}
        for bin_label, m in by_bin.items():
            row[f'within_bal_acc_t{bin_label}'] = m['balanced_acc']
            row[f'majority_baseline_t{bin_label}'] = m['majority_baseline']
            row[f'p_z1_t{bin_label}'] = m['p_z1']
        rows.append(row)
        print(f'  [{i}/{len(ckpts)}] {hd_mode}/seed{seed:02d}  across={across_acc:.4f}  '
              f'within_bal={within_acc:.4f}  ' +
              '  '.join(f'{k}=bal:{m["balanced_acc"]:.3f}/base:{m["majority_baseline"]:.3f}'
                       for k, m in by_bin.items()), flush=True)

    if not rows:
        raise SystemExit(f'no checkpoints matched hd_modes {a.hd_modes} under {a.runs}')

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f'\n{"hd_mode":<8s} {"n":>3s} {"across":>8s} {"within_bal":>10s}')
    for hd_mode in sorted({r['hd_mode'] for r in rows}):
        sub = [r for r in rows if r['hd_mode'] == hd_mode]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f'{hd_mode:<8s} {len(sub):>3d} {m("across_episode_phase_acc"):>8.4f} '
              f'{m("within_episode_balanced_acc"):>10.4f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
