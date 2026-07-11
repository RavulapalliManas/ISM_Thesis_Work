#!/usr/bin/env python3
"""Is spatial folding driven by how much information HD carries, or by whether
HD is invariant under the arena's symmetry group?

`axis` and `parity` carry exactly the same one bit about heading. Only `axis` is
C2-invariant. So in the C2 arena they make opposite predictions:

    information hypothesis:  fold(full) < fold(axis) ~ fold(parity) < fold(const)
    symmetry    hypothesis:  fold(full) ~ fold(parity) < fold(axis) ~ fold(const)

Same arena, same trajectories, same architecture. One 4x4 linear transform on the
heading block of `act` is the only thing that differs. Read out with the C4
isotypic spectrum (odd power) and orbit-phase decoding.

`s4` is included as a dose-response check: `axis` still breaks C4, so it should
sit strictly between `full` and `const` there, while in `s2` it should sit with
`const`.

All models in one invocation share the graph (F=7, T=200, and a single `--k`), so they
train as ONE vmap ensemble and the kernel-launch count is independent of how many there
are. A different `k` is a different graph, hence a different inductor compile: run one
invocation per horizon.

`--k` also drives the prediction-horizon dose-response. k=0 makes the target
`obs[anchor]` -- a pure autoencoder with no future to predict, so self-motion is
irrelevant to the objective and every HD encoding should fold alike. The HD-invariance
effect should therefore appear only for k > 0: symmetry is resolved by prediction, not
by observation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.hd_encodings import MODES, hd_matrix  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import (  # noqa: E402
    ANCHOR_SUBSAMPLE_N, ACT, CHECKPOINT_STEPS, DROPOUT_P, F, HIDDEN_INIT_SIGMA,
    HIDDEN_SIZE, K, NOISE_STD, OBS, PRED_OFFSET, T, _enable_tf32,
    _sample_anchor_idx, ensure_data)
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402
from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager  # noqa: E402

CONDITIONS = ('s1', 's2', 's4')

# Seeds are not spread uniformly. The decisive contrast -- `axis` vs `parity` in
# the C2 arena, matched at one bit and differing only in invariance -- lives in s2,
# and an exact two-sided Mann-Whitney at n=4 vs 4 bottoms out at p=0.029, which does
# not survive correction. n=6 buys p=0.0022. s4 carries the C4 dose-response
# (full > axis > const) as an ordered trend across 3 levels, so n=4 suffices. s1 is
# a specificity null -- HD mode should do nothing when there is no symmetry to fold
# -- and pools across the 4 modes to n=8.
DEFAULT_SEEDS = {'s1': 2, 's2': 6, 's4': 4}     # -> S = 8 + 24 + 16 = 48


def build_spec(seeds_per_cond, hd_modes):
    return [(c, hd, s) for c, n in seeds_per_cond.items() for hd in hd_modes for s in range(n)]


def init_seed(cond, hd, seed, conditions=None, hd_modes=MODES):
    """Deterministic, collision-free init seed per cell (up to 10 hd x 100 seeds).

    The condition index comes from the module-level CONDITIONS, never from the
    invocation's subset: otherwise `--seeds s4=2` alone would index s4 at 0 and hand it
    the same inits an s1-only run gets, so the "same" cell would be a different model
    depending on what else was in the run. `conditions` is accepted and ignored for
    call-site compatibility.
    """
    assert len(hd_modes) <= 10 and seed < 100
    return 1000 * CONDITIONS.index(cond) + 100 * list(hd_modes).index(hd) + seed


def build_models_k(init_seeds, device, k, hidden_size=HIDDEN_SIZE):
    """Like run_ensemble_sweep.build_models, but with the rollout horizon as a
    parameter. Each distinct k is a distinct graph, hence a distinct inductor
    compile -- so one k per ensemble, and the conditions x hd x seeds inside it.
    A distinct hidden_size is likewise a distinct graph (one compile per size).
    """
    models = []
    for s in init_seeds:
        torch.manual_seed(s)
        models.append(pRNN_th(obs_size=OBS, act_size=ACT, k=k, hidden_size=hidden_size,
                              cell=LayerNormRNNCellEager, dropp=DROPOUT_P, trunc=T,
                              neuralTimescale=2, predOffset=PRED_OFFSET,
                              hidden_init_sigma=HIDDEN_INIT_SIGMA).to(device))
    return models


def parse_seeds(pairs):
    out = {}
    for p in pairs:
        c, _, n = p.partition('=')
        if c not in CONDITIONS or not n.isdigit():
            raise ValueError(f'bad --seeds entry {p!r}, want e.g. s2=6')
        out[c] = int(n)
    return out


def sample_batches(stores, spec, B, hd_stack, hd_noise=0.0, learned_hd=False):
    """One independent batch per model, from that model's condition, with that
    model's HD transform applied to the heading block.

    The condition order MUST come from `spec`, not from a module constant: rows of
    `obs` are matched to models positionally, so any other order silently trains
    each model on the wrong arena. `spec` is grouped by condition by construction.

    `hd_noise` > 0 models an unreliable compass: before the encoding, each heading is
    rotated by a random +/-90 deg with probability `hd_noise`, so the network learns
    under a noisy head-direction signal. The corruption is applied to the true heading,
    not the encoded block, so it composes with any encoding.

    `learned_hd` replaces the absolute compass with ANGULAR VELOCITY: the per-step turn
    (none / +90 / 180 / -90 as a 4-way one-hot). The network is then given no absolute
    heading and must integrate the turns itself, so any heading it forms is defined only
    up to a global rotation -- an unanchored, path-integrated compass.
    """
    order = list(dict.fromkeys(c for c, _, _ in spec))
    obs_parts, act_parts = [], []
    for cond in order:
        n = sum(1 for c, _, _ in spec if c == cond)
        o, a = stores[cond].sample_parallel_batches(n, B)
        obs_parts.append(o)
        act_parts.append(a)
    obs = torch.cat(obs_parts, 0)
    act = torch.cat(act_parts, 0)                       # (S, B, T, 5)
    speed, hd = act[..., :1], act[..., 1:]
    if learned_hd:
        idx = hd.argmax(-1)                             # (S, B, T) absolute heading
        turn = (idx - torch.roll(idx, 1, dims=-1)) % 4  # per-step turn (angular velocity)
        turn[..., 0] = 0                                # no reference at the first step
        hd = torch.nn.functional.one_hot(turn, 4).to(hd.dtype)
        return obs, torch.cat([speed, hd], dim=-1)
    if hd_noise > 0:
        idx = hd.argmax(-1)                             # (S, B, T) true heading
        delta = torch.randint(0, 2, idx.shape, device=hd.device) * 2 - 1   # +/-1
        flip = torch.rand(idx.shape, device=hd.device) < hd_noise
        idx = torch.where(flip, (idx + delta) % 4, idx)
        hd = torch.nn.functional.one_hot(idx, 4).to(hd.dtype)
    hd = torch.einsum('sbtj,sij->sbti', hd, hd_stack)   # per-model 4x4 transform
    return obs, torch.cat([speed, hd], dim=-1)


def save_checkpoints(params, spec, out_root, step, meta):
    for i, (cond, hd, seed) in enumerate(spec):
        d = Path(out_root) / cond / hd / f'seed_{seed:02d}'
        d.mkdir(parents=True, exist_ok=True)
        name = 'ckpt_final.pt' if step == 'final' else f'ckpt_{step}.pt'
        torch.save({'step': step, 'model': ens.unstack_state_dict(params, i),
                    'meta': meta | {'condition': cond, 'hd_mode': hd, 'seed': seed}},
                   d / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/runs/hd_invariance')
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-traj', type=int, default=10_000)
    ap.add_argument('--k', type=int, default=K,
                    help='rollout horizon. k=0 is the autoencoder control (target == obs[anchor]).')
    ap.add_argument('--seeds', nargs='+', default=None,
                    help='per-condition seed counts, e.g. s1=2 s2=6 s4=4')
    ap.add_argument('--hd-noise', type=float, default=0.0,
                    help='per-step probability the compass is rotated +/-90 deg (unreliable HD)')
    ap.add_argument('--seed-offset', type=int, default=0,
                    help='shift seed ids, to extend an existing run with fresh seeds')
    ap.add_argument('--hd-modes', nargs='+', default=list(MODES))
    ap.add_argument('--learned-hd', action='store_true',
                    help='replace the oracle compass with angular velocity (turns); the network '
                         'integrates its own heading. Forces a single "learned" condition per arena.')
    ap.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE,
                    help='recurrent width; a distinct size is a distinct graph (separate compile)')
    ap.add_argument('--arena-size', type=int, default=18,
                    help='arena side length; obs stays FxF so the graph is unchanged, only the data differs')
    ap.add_argument('--dataset-workers', type=int, default=64)
    ap.add_argument('--compile', default='default', choices=['default', 'none'])
    ap.add_argument('--log-every', type=int, default=200)
    a = ap.parse_args()

    device = torch.device('cuda')
    _enable_tf32(device)
    k = a.k
    hidden_size = a.hidden_size
    if a.learned_hd:
        a.hd_modes = ['learned']        # one integrated-compass condition per arena
    seeds = parse_seeds(a.seeds) if a.seeds else dict(DEFAULT_SEEDS)
    spec = [(c, h, s + a.seed_offset) for c, h, s in build_spec(seeds, a.hd_modes)]
    conditions = list(seeds)
    S, B, T_k = len(spec), a.batch_size, T - k
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)

    print(f'device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'models : S={S}  seeds {seeds}  x {len(a.hd_modes)} hd modes')
    print(f'hd     : {a.hd_modes}')
    print(f'config : B={B} T={T} k={k} hidden={hidden_size} arena={a.arena_size} '
          f'steps={a.n_steps} compile={a.compile}\n', flush=True)

    stores = {}
    for cond in conditions:
        d = ensure_data(cond, a.data_root, a.n_traj, a.dataset_workers, size=a.arena_size)
        stores[cond] = PackedTrajectoryStore(str(d), device=device)
        print(f'  {cond}: {len(stores[cond])} trajectories', flush=True)

    if a.learned_hd:
        hd_stack = torch.eye(4, device=device).expand(S, 4, 4).contiguous()  # unused; see sample_batches
    else:
        hd_stack = torch.stack([hd_matrix(hd, device=device) for _, hd, _ in spec])   # (S,4,4)

    # Distinct, deterministic init per cell. NOT hash() -- that is salted per
    # process, so the run would not reproduce across restarts.
    models = build_models_k([init_seed(c, hd, s, conditions, a.hd_modes)
                            for c, hd, s in spec], device, k, hidden_size)

    obs, act = sample_batches(stores, spec, B, hd_stack, learned_hd=a.learned_hd)
    state = torch.rand(S, B, hidden_size, device=device) * HIDDEN_INIT_SIGMA
    nm = NOISE_STD * torch.randn(S, B, T, hidden_size, device=device)
    nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, hidden_size, device=device)
    aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)
    for i, m in enumerate(models):
        m.eval()
        ens.warm_buffers(m, obs[i], act[i], anchor_idx=aidx, state=state[i],
                         noise_main=nm[i], noise_roll=nr[i])
        m.train()

    params, buffers, base = ens.stack_models(models)
    base.train()
    leaves = ens.as_leaves(params)
    opt = ens.stacked_rmsprop(leaves, batch_size=B)
    gfn = ens.make_loss_and_grad_fn(base, buffers, randomness='different')
    if a.compile != 'none':
        gfn = torch.compile(gfn, mode=a.compile)

    meta = {'obs_size': OBS, 'k': k, 'trunc': T, 'hidden_size': hidden_size, 'F': F,
            'arena_size': a.arena_size, 'hd_noise': a.hd_noise, 'learned_hd': a.learned_hd,
            'batch_size': B, 'noise_std': NOISE_STD, 'dropout_p': DROPOUT_P,
            'hidden_init_sigma': HIDDEN_INIT_SIGMA, 'pred_offset': PRED_OFFSET,
            'n_steps': a.n_steps, 'runner': 'ensemble_vmap_hd_invariance'}
    log = {f'{c}/{h}/seed_{s:02d}': {'steps': [], 'loss': []} for c, h, s in spec}

    print('compiling + training...', flush=True)
    t0 = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = sample_batches(stores, spec, B, hd_stack, a.hd_noise, learned_hd=a.learned_hd)
        state = torch.rand(S, B, hidden_size, device=device) * HIDDEN_INIT_SIGMA
        nm = NOISE_STD * torch.randn(S, B, T, hidden_size, device=device)
        nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, hidden_size, device=device)
        aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)

        grads, losses = gfn(leaves, obs, act, state, nm, nr, aidx)
        grads = {k: v.clone() for k, v in grads.items()}
        ens.clip_per_slice(grads, max_norm=1.0)
        opt.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, grads)
        opt.step()

        if step == 1:
            print(f'  first step (incl. compile) {time.perf_counter()-t0:.0f}s  '
                  f'peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
        if step % a.log_every == 0 or step == 1:
            l = losses.detach().cpu().tolist()
            for (c, h, s), v in zip(spec, l):
                log[f'{c}/{h}/seed_{s:02d}']['steps'].append(step)
                log[f'{c}/{h}/seed_{s:02d}']['loss'].append(v)
            if step % (a.log_every * 25) == 0:
                el = time.perf_counter() - t0
                r = step / el
                print(f'  step {step:>6}/{a.n_steps}  loss {sum(l)/len(l):.5f}  '
                      f'{1000/r:.1f} ms/step  eta {(a.n_steps-step)/r/60:.0f} min', flush=True)
        if step in CHECKPOINT_STEPS:
            save_checkpoints(leaves, spec, out_root, step, meta)
            print(f'  [ckpt] {step}', flush=True)

    save_checkpoints(leaves, spec, out_root, 'final', meta)
    for c, h, s in spec:
        d = out_root / c / h / f'seed_{s:02d}'
        with open(d / 'training_log.json', 'w') as f:
            json.dump(log[f'{c}/{h}/seed_{s:02d}'] | {'meta': meta}, f)

    tot = time.perf_counter() - t0
    print(f'\nDONE  {a.n_steps} steps in {tot/60:.1f} min '
          f'({tot/a.n_steps*1000:.1f} ms/ensemble-step, {tot/a.n_steps/S*1000:.2f} ms/model-step)',
          flush=True)


if __name__ == '__main__':
    main()
