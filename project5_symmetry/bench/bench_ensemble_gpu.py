#!/usr/bin/env python3
"""Calibrate the vmap ensemble on a real GPU.

Answers the one question the whole compute plan rests on: what does an ensemble
step actually cost, and is the launch count really amortised across S models?

Prediction to falsify (from CPU-measured launch counts x H100 launch latency):
  * sequential:  ~63 ms per model-step
  * vmap:        ~77 ms per ENSEMBLE-step, independent of S
  => ~21x at S=26, more with inductor fusion / CUDA graphs.

Run:  python bench_ensemble_gpu.py            # full matrix
      python bench_ensemble_gpu.py --quick    # S in {1,8}, no compile
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.training import ensemble as ens  # noqa: E402
from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager  # noqa: E402

OBS, ACT, HID, K, A = 147, 5, 500, 5, 32
DEV = torch.device('cuda')


def build_models(S, T):
    ms = []
    for i in range(S):
        torch.manual_seed(i)
        ms.append(pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HID,
                          cell=LayerNormRNNCellEager, dropp=0.15, trunc=T,
                          neuralTimescale=2, predOffset=0,
                          hidden_init_sigma=0.1).to(DEV))
    return ms


def make_inputs(S, B, T):
    torch.manual_seed(1234)
    return (torch.randn(S, B, T + 1, OBS, device=DEV),
            torch.randn(S, B, T, ACT, device=DEV),
            torch.rand(S, B, HID, device=DEV) * 0.1,
            torch.randn(S, B, T, HID, device=DEV) * 0.03,
            torch.randn(S, B, K, A, HID, device=DEV) * 0.03,
            torch.sort(torch.randperm(T - K, device=DEV)[:A]).values)


def timed(step, iters, warmup):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), min(ts)


def bench_sequential(B, T, iters, warmup):
    """What train_parallel_seeds does: one model at a time."""
    import torch.nn.functional as F
    m = build_models(1, T)[0].train()
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-3)
    obs, act, state, nm, nr, aidx = make_inputs(1, B, T)

    def step():
        opt.zero_grad(set_to_none=True)
        pred, _, tgt = m(obs[0], act[0], anchor_idx=aidx, state=state[0],
                         noise_main=nm[0], noise_roll=nr[0])
        F.mse_loss(pred, tgt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    med, lo = timed(step, iters, warmup)
    del m, opt
    torch.cuda.empty_cache()
    return med, lo


def bench_ensemble(S, B, T, compile_mode, iters, warmup):
    models = build_models(S, T)
    obs, act, state, nm, nr, aidx = make_inputs(S, B, T)
    for i, m in enumerate(models):
        m.eval()
        ens.warm_buffers(m, obs[i], act[i], anchor_idx=aidx, state=state[i],
                         noise_main=nm[i], noise_roll=nr[i])
        m.train()
    params, buffers, base = ens.stack_models(models)
    base.train()
    leaves = ens.as_leaves(params)
    opt = ens.stacked_rmsprop(leaves, batch_size=B)
    gfn = ens.make_grad_fn(base, buffers, randomness='different')
    if compile_mode:
        gfn = torch.compile(gfn, mode=compile_mode)

    def step():
        g = gfn(leaves, obs, act, state, nm, nr, aidx)
        g = {k: v.clone() for k, v in g.items()}
        ens.clip_per_slice(g, max_norm=1.0)
        opt.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, g)
        opt.step()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step()                                   # includes compile on first call
    setup = time.perf_counter() - t0
    med, lo = timed(step, iters, warmup)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del models, params, leaves, opt, gfn, base
    torch.cuda.empty_cache()
    return med, lo, peak, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    print(f'device : {torch.cuda.get_device_name(0)}   torch {torch.__version__}')
    cc = torch.cuda.get_device_capability(0)
    print(f'arch   : sm_{cc[0]}{cc[1]}   supported={("sm_%d%d" % cc) in torch.cuda.get_arch_list()}')
    print(f'shape  : B={a.batch} T={a.T} hidden={HID} k={K} anchors={A}\n', flush=True)

    seq_med, seq_lo = bench_sequential(a.batch, a.T, a.iters, a.warmup)
    print(f'SEQUENTIAL baseline (1 model, what the codebase does today)')
    print(f'  {seq_med*1000:.1f} ms per model-step   (min {seq_lo*1000:.1f})\n', flush=True)

    S_list = (1, 8) if a.quick else (1, 4, 8, 17, 26)
    modes = [None] if a.quick else [None, 'default', 'reduce-overhead']

    for mode in modes:
        label = mode or 'no compile'
        print(f'--- vmap ensemble, {label} ---')
        print(f"{'S':>4}{'ms/ens-step':>14}{'ms/model-step':>16}{'speedup':>10}"
              f"{'peak GB':>10}{'setup s':>10}")
        for S in S_list:
            try:
                med, lo, peak, setup = bench_ensemble(S, a.batch, a.T, mode, a.iters, a.warmup)
                print(f'{S:>4}{med*1000:>14.1f}{med*1000/S:>16.2f}'
                      f'{seq_med/(med/S):>10.1f}x{peak:>10.2f}{setup:>10.1f}', flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f'{S:>4}   OOM'); torch.cuda.empty_cache(); break
            except Exception as e:
                print(f'{S:>4}   FAILED: {type(e).__name__}: {str(e)[:60]}', flush=True)
                torch.cuda.empty_cache()
                if S == S_list[0]:
                    break
        print(flush=True)

    print('projection: 26-model group, ms/ens-step x steps')
    print(f"{'ms/step':>10}{'30k (min)':>12}{'80k (min)':>12}")
    for msv in (77, 40, 20, 10):
        print(f'{msv:>10}{30000*msv/1000/60:>12.1f}{80000*msv/1000/60:>12.1f}')


if __name__ == '__main__':
    main()
