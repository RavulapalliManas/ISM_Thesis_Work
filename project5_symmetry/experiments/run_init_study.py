#!/usr/bin/env python3
"""What does the initialisation preconfigure, and does preconfiguration buy learning speed?

`pRNN_th` starts from W = U(-1/sqrt(H), 1/sqrt(H)) + (1 - 1/tau) * I -- a leaky-integrator
prior with tau fixed at 2 throughout the paper. This sweep varies the identity component,
the gain of the random part, and its structure (see training/inits.py), holding everything
else fixed. The optimiser is untouched: its learning rates depend on hidden size, not on
the init's scale.

Run on `annulus` (b1 = 1) so that all three readouts are available on the same models:

    speed     steps to reach a loss threshold, from the training log
    topology  b1 of the population manifold -- AT STEP 0 and across training
    geometry  Spearman(neural distance, geodesic distance), and position decoding

The claim to test: an initialisation whose step-0 manifold already carries the arena's
loop learns the geometry faster. That is the mechanistic form of Guardamagna et al. (2026),
where toroidal manifolds appear at P10 -- before eye opening and exploration -- and are
only later "anchored to the external world through experience-dependent plasticity".

Init variants share the computation graph, so all 8 x seeds train as ONE vmap ensemble.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.topology_arenas import EXPECTED_B1, betti  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import (  # noqa: E402
    ACT, ANCHOR_SUBSAMPLE_N, DROPOUT_P, F, HIDDEN_INIT_SIGMA, HIDDEN_SIZE, K,
    NOISE_STD, OBS, PRED_OFFSET, T, _enable_tf32, _sample_anchor_idx)
from project5_symmetry.experiments.run_hd_invariance import build_models_k  # noqa: E402
from project5_symmetry.experiments.run_topology import (CHECKPOINT_STEPS,  # noqa: E402
                                                        ensure_data)
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402
from project5_symmetry.training.inits import VARIANTS, apply_init, spectral_radius  # noqa: E402


def init_seed(variant, seed):
    """The seed does NOT depend on the variant -- that is the point.

    Every arm at a given seed is built from the same torch.manual_seed, so W_in, W_out and
    bias are IDENTICAL across variants, and `recurrent_weight` draws the same base random
    matrix R. The arms then differ only in the recurrent matrix: `gain_lo` is literally
    0.5 * R + 0.5*I where baseline is R + 0.5*I, and `tau8` is the same R with a different
    diagonal. A paired design, not merely a matched one.
    """
    return seed


def save_checkpoints(params, spec, out_root, step, meta, layout):
    for i, (variant, seed) in enumerate(spec):
        tau, gain, struct = VARIANTS[variant]
        d = Path(out_root) / variant / f'seed_{seed:02d}'
        d.mkdir(parents=True, exist_ok=True)
        name = 'ckpt_final.pt' if step == 'final' else f'ckpt_{step}.pt'
        torch.save({'step': step, 'model': ens.unstack_state_dict(params, i),
                    'meta': meta | {'variant': variant, 'seed': seed, 'layout': layout,
                                    'tau': tau, 'gain': gain, 'struct': struct,
                                    'b1': EXPECTED_B1[layout]}}, d / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/runs/init_study')
    ap.add_argument('--data-root', default='/root/data/topology')
    ap.add_argument('--layout', default='annulus')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-traj', type=int, default=10_000)
    ap.add_argument('--n-seeds', type=int, default=4)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--dataset-workers', type=int, default=16)
    ap.add_argument('--compile', default='default', choices=['default', 'none'])
    ap.add_argument('--log-every', type=int, default=200)
    a = ap.parse_args()

    device = torch.device('cuda')
    _enable_tf32(device)
    k = a.k
    spec = [(v, s) for v in a.variants for s in range(a.n_seeds)]
    S, B, T_k = len(spec), a.batch_size, T - k
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    b0, b1 = betti(a.layout)

    print(f'device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'arena  : {a.layout}  b0={b0} b1={b1}')
    print(f'models : S={S} = {len(a.variants)} inits x {a.n_seeds} seeds')
    print(f'config : B={B} T={T} k={k} steps={a.n_steps}')
    print(f'ckpts  : {CHECKPOINT_STEPS}\n', flush=True)

    d = ensure_data(a.layout, a.data_root, a.n_traj, a.dataset_workers)
    store = PackedTrajectoryStore(str(d), device=device)
    print(f'  {a.layout}: {len(store)} trajectories', flush=True)

    models = build_models_k([init_seed(v, s) for v, s in spec], device, k)
    for m, (v, s) in zip(models, spec):
        apply_init(m, v, init_seed(v, s))
    # paired design: within a seed only W differs
    for s in range(a.n_seeds):
        arms = [i for i, (_, ss) in enumerate(spec) if ss == s]
        for i in arms[1:]:
            for name in ('W_in', 'W_out', 'bias'):
                assert torch.equal(getattr(models[arms[0]], name), getattr(models[i], name)), \
                    f'seed {s}: {name} differs across init variants; the design is not paired'
        assert not torch.equal(models[arms[0]].W, models[arms[-1]].W)
    for v in a.variants:
        i = next(j for j, (vv, _) in enumerate(spec) if vv == v)
        W = models[i].W.detach().cpu().numpy()
        print(f'  {v:<10s} diag={W.diagonal().mean():+.3f}  rho={spectral_radius(W):.3f}',
              flush=True)

    def sample(S_, B_):
        return store.sample_parallel_batches(S_, B_)

    obs, act = sample(S, B)
    state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
    nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
    nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
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

    meta = {'obs_size': OBS, 'k': k, 'trunc': T, 'hidden_size': HIDDEN_SIZE, 'F': F,
            'batch_size': B, 'noise_std': NOISE_STD, 'dropout_p': DROPOUT_P,
            'hidden_init_sigma': HIDDEN_INIT_SIGMA, 'pred_offset': PRED_OFFSET,
            'n_steps': a.n_steps, 'runner': 'ensemble_vmap_init_study'}
    log = {f'{v}/seed_{s:02d}': {'steps': [], 'loss': []} for v, s in spec}

    save_checkpoints(leaves, spec, out_root, 0, meta, a.layout)
    print('  [ckpt] 0  (the initialisation itself)', flush=True)

    print('compiling + training...', flush=True)
    t0 = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = sample(S, B)
        state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
        nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
        nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
        aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)

        grads, losses = gfn(leaves, obs, act, state, nm, nr, aidx)
        grads = {kk: v.clone() for kk, v in grads.items()}
        ens.clip_per_slice(grads, max_norm=1.0)
        opt.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, grads)
        opt.step()

        if step == 1:
            print(f'  first step (incl. compile) {time.perf_counter()-t0:.0f}s  '
                  f'peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
        if step % a.log_every == 0 or step == 1:
            l_ = losses.detach().cpu().tolist()
            for (v, s), val in zip(spec, l_):
                log[f'{v}/seed_{s:02d}']['steps'].append(step)
                log[f'{v}/seed_{s:02d}']['loss'].append(val)
            if step % 5000 == 0:
                el = time.perf_counter() - t0
                per = {v: sum(log[f'{v}/seed_{s:02d}']['loss'][-1] for s in range(a.n_seeds))
                          / a.n_seeds for v in a.variants}
                best = min(per, key=per.get)
                print(f'  step {step:>6}/{a.n_steps}  best {best}={per[best]:.5f}  '
                      f'{el/step*1000:.1f} ms/step  eta {(a.n_steps-step)*el/step/60:.0f} min',
                      flush=True)
        if step in CHECKPOINT_STEPS:
            save_checkpoints(leaves, spec, out_root, step, meta, a.layout)
            print(f'  [ckpt] {step}', flush=True)

    save_checkpoints(leaves, spec, out_root, 'final', meta, a.layout)
    for v, s in spec:
        with open(out_root / v / f'seed_{s:02d}' / 'training_log.json', 'w') as f:
            json.dump(log[f'{v}/seed_{s:02d}'] | {'meta': meta | {'variant': v, 'seed': s}}, f)

    tot = time.perf_counter() - t0
    print(f'\nDONE  {a.n_steps} steps in {tot/60:.1f} min '
          f'({tot/a.n_steps*1000:.1f} ms/ensemble-step, {tot/a.n_steps/S*1000:.2f} ms/model-step)',
          flush=True)


if __name__ == '__main__':
    main()
