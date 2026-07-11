#!/usr/bin/env python3
"""Training-step benchmark for the pRNN.

Reports median ms/step and samples/s so "ultra optimized" is a measured number.

    python project5_symmetry/bench/bench_step.py                       # defaults
    python project5_symmetry/bench/bench_step.py --batch 16 32 64 128  # batch sweep
    python project5_symmetry/bench/bench_step.py --device cuda --compile

Notes
-----
* The workload is launch/dispatch-bound, not FLOP-bound: a single cell call moves
  ~9 MFLOP. If ms/step grows much slower than the batch size, there is free
  throughput in a bigger batch (or in batching seeds).
* --compile only helps once the eager cell is in use; torch.compile cannot trace
  into the legacy jit.ScriptModule cell (it captures zero graphs). On CUDA,
  mode='reduce-overhead' additionally captures CUDA graphs.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager, LayerNormRNNCellScript  # noqa: E402

OBS, ACT = 147, 5
CELLS = {'eager': LayerNormRNNCellEager, 'script': LayerNormRNNCellScript}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench(cell, batch, T, hidden, k, anchors, device, compile_mode, iters, warmup):
    torch.manual_seed(0)
    model = pRNN_th(obs_size=OBS, act_size=ACT, k=k, hidden_size=hidden, cell=CELLS[cell],
                    dropp=0.15, trunc=T, neuralTimescale=2, predOffset=0,
                    hidden_init_sigma=0.1).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = torch.randn(batch, T + 1, OBS, device=device)
    act = torch.randn(batch, T, ACT, device=device)
    aidx = torch.sort(torch.randperm(T - k, device=device)[:min(anchors, T - k)]).values

    # Pre-warm the cached Toeplitz buffers before any compile/capture.
    with torch.no_grad():
        model(obs, act, anchor_idx=aidx)

    if compile_mode:
        model = torch.compile(model, mode=compile_mode)

    def step():
        opt.zero_grad(set_to_none=True)
        y, _, tgt = model(obs, act, anchor_idx=aidx)
        ((y - tgt) ** 2).mean().backward()
        opt.step()

    for _ in range(warmup):
        step()
    sync(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['script', 'eager'], choices=list(CELLS))
    ap.add_argument('--batch', nargs='+', type=int, default=[16])
    ap.add_argument('--T', type=int, default=200)
    ap.add_argument('--hidden', type=int, default=500)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--anchors', type=int, default=32)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--compile', dest='compile_mode', nargs='?', const='reduce-overhead',
                    default=None, help="torch.compile mode (default 'reduce-overhead')")
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--steps-budget', type=int, default=80000,
                    help='steps per seed, for the wall-clock projection')
    a = ap.parse_args()

    device = torch.device(a.device)
    print(f'device={device}  T={a.T}  hidden={a.hidden}  k={a.k}  anchors={a.anchors}  '
          f'compile={a.compile_mode or "off"}\n')
    print(f"{'cell':>7}{'B':>6}{'ms/step':>11}{'ms/sample':>12}{'samples/s':>12}"
          f"{'h/seed @' + str(a.steps_budget // 1000) + 'k':>16}")

    baseline = None
    for cell in a.cells:
        for b in a.batch:
            t = bench(cell, b, a.T, a.hidden, a.k, a.anchors, device,
                      a.compile_mode, a.iters, a.warmup)
            if baseline is None:
                baseline = t
            print(f'{cell:>7}{b:>6}{t*1000:>11.1f}{t*1000/b:>12.2f}{b/t:>12.1f}'
                  f'{a.steps_budget*t/3600:>16.2f}')

    print('\nIf ms/step grows much slower than B, the step is launch-bound and a larger')
    print('batch (or seed-batching via bmm) buys throughput almost for free.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
