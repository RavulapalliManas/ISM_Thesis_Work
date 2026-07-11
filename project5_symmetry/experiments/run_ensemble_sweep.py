#!/usr/bin/env python3
"""Train the whole symmetry sweep as ONE batched GPU job.

Replaces `train_parallel_seeds`, which stepped S models one after another in a
Python loop (so S seeds cost S x the time). Here the S models live on a leading
tensor dimension and every per-model `mm` becomes a batched `bmm`, so the kernel
launch count is independent of S.

Measured on an RTX 5090 (B=16, T=200, hidden=500):
    sequential                     73.6 ms per model-step
    vmap S=17, no compile         168.2 ms per ENSEMBLE-step  (9.90 ms/model)
    vmap S=17 + inductor           30.9 ms per ENSEMBLE-step  (1.82 ms/model, 40.5x)
`mode='reduce-overhead'` (CUDA graphs) measured *worse* (65.1 ms) -- use 'default'.

Fidelity to the original runs
-----------------------------
Mirrors `run_symmetry_sweep.py` (recovered from git; the surviving
`experiments/run_sweep.py` is stale and, among other things, builds s4 with
`use_landmarks=False`, which is wrong):
  * arenas: SymmetryArena(square, 18, U=0, F=7, seed=0, use_landmarks=True,
    symmetry_condition=cond) -- one dataset per condition, arena seed fixed at 0
  * P0_CFG: k=5, T=200, n_traj=10000, B=8, n_steps=80000
  * RMSprop with four per-group LRs and bias-only weight decay (_build_optimizer)
  * per-model gradient clipping at max_norm=1.0
  * NOISE_STD=0.03, DROPOUT_P=0.15, hidden init U(0, 0.1), 32 shared anchors

Randomness is pre-drawn per model and passed in (state, noise_main, noise_roll)
so the graph is pure; dropout stays inside the model under
`vmap(randomness='different')`, because `forward` builds the prediction target
from the RAW obs and only then drops the input -- pre-masking obs corrupts it.

Checkpoints are written per seed, unstacked, in the exact layout a plain
`pRNN_th` loads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Persist the inductor cache across runs (the compile costs ~7 min). Importing this
# module must not point compile at a path that may not exist -- the GPU box sets
# TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache explicitly in the launch command.
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',
                      str(Path.home() / '.cache' / 'prnn_inductor'))

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.arena import PixelObsWrapper, SymmetryArena  # noqa: E402
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402
from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager  # noqa: E402

# Mirrored from project5_symmetry/training/train.py. NOT imported: train.py pulls
# in tensorboard and evaluation/metrics.py (pynapple), neither of which a GPU
# training box needs. tests/project5_symmetry/test_ensemble.py pins these against
# train.py so they cannot drift.
HIDDEN_SIZE = 500
NOISE_STD = 0.03
DROPOUT_P = 0.15
HIDDEN_INIT_SIGMA = 0.1
PRED_OFFSET = 0
ANCHOR_SUBSAMPLE_N = 32
CHECKPOINT_STEPS = {5000, 10000, 20000, 40000, 60000, 80000}

# The 17 seeds behind the paper: s1 x5, s2 x5, s4 x7.
SWEEP = [('s1', s) for s in range(5)] + [('s2', s) for s in range(5)] + [('s4', s) for s in range(7)]
F, K, T, OBS, ACT = 7, 5, 200, 147, 5


def _enable_tf32(device):
    if device.type != 'cuda':
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def _sample_anchor_idx(T_k: int, device, anchor_subsample_n: int):
    if anchor_subsample_n <= 0 or anchor_subsample_n >= T_k:
        return None
    idx = torch.randperm(T_k, device=device)[:anchor_subsample_n]
    return torch.sort(idx).values


def build_env(condition, size=18):
    return PixelObsWrapper(
        SymmetryArena(shape='square', size=size, U=0, F=F, seed=0,
                      use_landmarks=True, symmetry_condition=condition),
        tile_size=1)


def ensure_data(condition, data_root, n_traj, workers, size=18):
    # Default size keeps the original path (data_root/cond) for backward compat;
    # a non-default arena size gets its own dir so size-N data never collides
    # with the size-18 sweep in the same data_root.
    d = Path(data_root) / (condition if size == 18 else f'{condition}_s{size}')
    generate_dataset(build_env(condition, size), n_traj=n_traj, T=T, out_dir=str(d),
                     n_workers=workers, desc=f'{condition}(size {size}) trajectories')
    return d


def build_models(specs, device):
    models = []
    for _, seed in specs:
        torch.manual_seed(seed)
        models.append(pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HIDDEN_SIZE,
                              cell=LayerNormRNNCellEager, dropp=DROPOUT_P, trunc=T,
                              neuralTimescale=2, predOffset=PRED_OFFSET,
                              hidden_init_sigma=HIDDEN_INIT_SIGMA).to(device))
    return models


def sample_batches(stores, specs, B, device):
    """One independent batch per model, drawn from that model's condition."""
    obs_parts, act_parts = [], []
    for cond in ('s1', 's2', 's4'):
        n = sum(1 for c, _ in specs if c == cond)
        if not n:
            continue
        o, a = stores[cond].sample_parallel_batches(n, B)
        obs_parts.append(o)
        act_parts.append(a)
    return torch.cat(obs_parts, 0), torch.cat(act_parts, 0)


def save_checkpoints(params, specs, out_root, step, meta):
    for i, (cond, seed) in enumerate(specs):
        d = Path(out_root) / cond / f'seed_{seed:02d}'
        d.mkdir(parents=True, exist_ok=True)
        name = 'ckpt_final.pt' if step == 'final' else f'ckpt_{step}.pt'
        torch.save({'step': step, 'model': ens.unstack_state_dict(params, i), 'meta': meta},
                   d / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/runs/symmetry_sweep')
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)      # P0_CFG.B
    ap.add_argument('--n-traj', type=int, default=10_000)     # P0_CFG.n_traj
    ap.add_argument('--dataset-workers', type=int, default=32)
    ap.add_argument('--compile', default='default', choices=['default', 'none'])
    ap.add_argument('--log-every', type=int, default=200)
    a = ap.parse_args()

    device = torch.device('cuda')
    _enable_tf32(device)                      # matches train.py on CUDA
    B, S = a.batch_size, len(SWEEP)
    T_k = T - K
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'models : S={S}  {SWEEP}')
    print(f'config : B={B} T={T} k={K} steps={a.n_steps} compile={a.compile}\n', flush=True)

    t0 = time.perf_counter()
    stores = {}
    for cond in ('s1', 's2', 's4'):
        d = ensure_data(cond, a.data_root, a.n_traj, a.dataset_workers)
        stores[cond] = PackedTrajectoryStore(str(d), device=device)
        print(f'  {cond}: {len(stores[cond])} trajectories on device', flush=True)
    print(f'data ready in {time.perf_counter()-t0:.0f}s\n', flush=True)

    models = build_models(SWEEP, device)
    obs, act = sample_batches(stores, SWEEP, B, device)
    state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
    nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
    nr = NOISE_STD * torch.randn(S, B, K, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
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

    meta = {'obs_size': OBS, 'k': K, 'trunc': T, 'hidden_size': HIDDEN_SIZE,
            'batch_size': B, 'noise_std': NOISE_STD, 'dropout_p': DROPOUT_P,
            'hidden_init_sigma': HIDDEN_INIT_SIGMA, 'n_steps': a.n_steps,
            'anchor_subsample_n': ANCHOR_SUBSAMPLE_N, 'runner': 'ensemble_vmap'}
    log = {f'{c}/seed_{s:02d}': {'steps': [], 'loss': []} for c, s in SWEEP}

    print('compiling + training...', flush=True)
    t_start = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = sample_batches(stores, SWEEP, B, device)
        state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
        nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
        nr = NOISE_STD * torch.randn(S, B, K, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
        aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)

        grads, losses = gfn(leaves, obs, act, state, nm, nr, aidx)
        grads = {k: v.clone() for k, v in grads.items()}
        ens.clip_per_slice(grads, max_norm=1.0)
        opt.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, grads)
        opt.step()

        if step == 1:
            print(f'  first step done (incl. compile) in '
                  f'{time.perf_counter()-t_start:.0f}s', flush=True)
        if step % a.log_every == 0 or step == 1:
            l = losses.detach().cpu().tolist()
            for (c, s), v in zip(SWEEP, l):
                log[f'{c}/seed_{s:02d}']['steps'].append(step)
                log[f'{c}/seed_{s:02d}']['loss'].append(v)
            if step % (a.log_every * 25) == 0 or step == 1:
                el = time.perf_counter() - t_start
                rate = step / el
                print(f'  step {step:>6}/{a.n_steps}  loss {sum(l)/len(l):.5f}  '
                      f'{1000/rate:.1f} ms/step  eta {(a.n_steps-step)/rate/60:.0f} min',
                      flush=True)

        if step in CHECKPOINT_STEPS:
            save_checkpoints(leaves, SWEEP, out_root, step, meta)
            print(f'  [ckpt] step {step}', flush=True)

    save_checkpoints(leaves, SWEEP, out_root, 'final', meta)
    for (c, s) in SWEEP:
        d = out_root / c / f'seed_{s:02d}'
        with open(d / 'training_log.json', 'w') as f:
            json.dump(log[f'{c}/seed_{s:02d}'] | {'meta': meta}, f)

    total = time.perf_counter() - t_start
    print(f'\nDONE  {a.n_steps} steps in {total/60:.1f} min '
          f'({total/a.n_steps*1000:.1f} ms/ensemble-step, '
          f'{total/a.n_steps/S*1000:.2f} ms/model-step)', flush=True)


if __name__ == '__main__':
    main()
