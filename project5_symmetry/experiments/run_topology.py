#!/usr/bin/env python3
"""Does a predictive code recover the environment's TOPOLOGY before its GEOMETRY?

Trains one vmap ensemble over arenas x seeds, saving LOG-SPACED checkpoints starting at
step 0. The developmental axis is training time; the readouts (see analysis/run_tda.py)
are, at each checkpoint:

    topology :  persistent H1 of the position-conditioned population manifold.
                Does the number of long-lived H1 bars equal the arena's b1?
    geometry :  how faithfully neural distance tracks geodesic distance in the arena,
                and how well position decodes.

The prediction under test: the topology score saturates at an earlier training step than
the geometry score.

THE CONTROL THAT DECIDES THE INTERPRETATION is step 0. A randomly initialised RNN driven
by observations from the arena may already trace the arena's loop, in which case topology
is *inherited from the input*, not learned -- which is itself the model analogue of
Guardamagna et al. (2026), where toroidal manifolds appear at P10, before eye opening and
exploration ("spatial representations are preconfigured and later anchored to the external
world through experience-dependent plasticity"). Either outcome is a result; failing to
measure step 0 would make the claim unfalsifiable.

All arenas share the graph (F=7, k, T=200), so they train as ONE ensemble.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.environments.topology_arenas import (EXPECTED_B1,  # noqa: E402
                                                            LAYOUTS, betti,
                                                            make_topology_env)
from project5_symmetry.experiments.run_ensemble_sweep import (  # noqa: E402
    ACT, ANCHOR_SUBSAMPLE_N, DROPOUT_P, F, HIDDEN_INIT_SIGMA, HIDDEN_SIZE, K,
    NOISE_STD, OBS, PRED_OFFSET, T, _enable_tf32, _sample_anchor_idx)
from project5_symmetry.experiments.run_hd_invariance import build_models_k  # noqa: E402
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402

# Log-spaced, dense where the interesting thing is claimed to happen (early).
CHECKPOINT_STEPS = (0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 80000)


def ensure_data(layout, data_root, n_traj, workers):
    """`env_factory` is mandatory here. Without it `generate_dataset` rebuilds the env from
    kwargs inside each worker and downcasts TopologyArena to a plain SymmetryArena -- the
    walls vanish and every arena becomes the same open box."""
    d = Path(data_root) / layout
    generate_dataset(make_topology_env(layout, F=F, seed=0), n_traj=n_traj, T=T,
                     out_dir=str(d), n_workers=workers, desc=f'{layout} trajectories',
                     env_factory=make_topology_env,
                     factory_kwargs={'layout': layout, 'F': F, 'seed': 0})
    return d


def init_seed(layout, seed):
    return 1000 * LAYOUTS.index(layout) + seed


def sample_batches(stores, spec, B):
    """Order comes from `spec`: obs rows are matched to models positionally."""
    order = list(dict.fromkeys(l for l, _ in spec))
    obs_parts, act_parts = [], []
    for layout in order:
        n = sum(1 for l, _ in spec if l == layout)
        o, a = stores[layout].sample_parallel_batches(n, B)
        obs_parts.append(o)
        act_parts.append(a)
    return torch.cat(obs_parts, 0), torch.cat(act_parts, 0)


def save_checkpoints(params, spec, out_root, step, meta):
    for i, (layout, seed) in enumerate(spec):
        d = Path(out_root) / layout / f'seed_{seed:02d}'
        d.mkdir(parents=True, exist_ok=True)
        name = 'ckpt_final.pt' if step == 'final' else f'ckpt_{step}.pt'
        torch.save({'step': step, 'model': ens.unstack_state_dict(params, i),
                    'meta': meta | {'layout': layout, 'seed': seed,
                                    'b1': EXPECTED_B1[layout]}}, d / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/runs/topology')
    ap.add_argument('--data-root', default='/root/data/topology')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-traj', type=int, default=10_000)
    ap.add_argument('--n-seeds', type=int, default=4)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--layouts', nargs='+', default=list(LAYOUTS))
    ap.add_argument('--dataset-workers', type=int, default=16)
    ap.add_argument('--compile', default='default', choices=['default', 'none'])
    ap.add_argument('--log-every', type=int, default=200)
    a = ap.parse_args()

    device = torch.device('cuda')
    _enable_tf32(device)
    k = a.k
    spec = [(l, s) for l in a.layouts for s in range(a.n_seeds)]
    S, B, T_k = len(spec), a.batch_size, T - k
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)

    print(f'device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'models : S={S} = {len(a.layouts)} arenas x {a.n_seeds} seeds')
    for l in a.layouts:
        b0, b1 = betti(l)
        print(f'  {l:<12s} b0={b0} b1={b1}')
    print(f'config : B={B} T={T} k={k} steps={a.n_steps}')
    print(f'ckpts  : {CHECKPOINT_STEPS}\n', flush=True)

    stores = {}
    for l in a.layouts:
        d = ensure_data(l, a.data_root, a.n_traj, a.dataset_workers)
        stores[l] = PackedTrajectoryStore(str(d), device=device)
        print(f'  {l}: {len(stores[l])} trajectories', flush=True)

    models = build_models_k([init_seed(l, s) for l, s in spec], device, k)

    obs, act = sample_batches(stores, spec, B)
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
            'n_steps': a.n_steps, 'runner': 'ensemble_vmap_topology'}
    log = {f'{l}/seed_{s:02d}': {'steps': [], 'loss': []} for l, s in spec}

    # step 0: the untrained network. The control that decides the interpretation.
    save_checkpoints(leaves, spec, out_root, 0, meta)
    print('  [ckpt] 0  (random init)', flush=True)

    print('compiling + training...', flush=True)
    t0 = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = sample_batches(stores, spec, B)
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
            for (lay, s), v in zip(spec, l_):
                log[f'{lay}/seed_{s:02d}']['steps'].append(step)
                log[f'{lay}/seed_{s:02d}']['loss'].append(v)
            if step % 5000 == 0:
                el = time.perf_counter() - t0
                print(f'  step {step:>6}/{a.n_steps}  loss {sum(l_)/len(l_):.5f}  '
                      f'{el/step*1000:.1f} ms/step  eta {(a.n_steps-step)*el/step/60:.0f} min',
                      flush=True)
        if step in CHECKPOINT_STEPS:
            save_checkpoints(leaves, spec, out_root, step, meta)
            print(f'  [ckpt] {step}', flush=True)

    save_checkpoints(leaves, spec, out_root, 'final', meta)
    for l, s in spec:
        with open(out_root / l / f'seed_{s:02d}' / 'training_log.json', 'w') as f:
            json.dump(log[f'{l}/seed_{s:02d}'] | {'meta': meta}, f)

    tot = time.perf_counter() - t0
    print(f'\nDONE  {a.n_steps} steps in {tot/60:.1f} min '
          f'({tot/a.n_steps*1000:.1f} ms/ensemble-step, {tot/a.n_steps/S*1000:.2f} ms/model-step)',
          flush=True)


if __name__ == '__main__':
    main()
