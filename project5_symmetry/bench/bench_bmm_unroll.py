#!/usr/bin/env python3
"""Ceiling test: what does the batched recurrence cost if we skip vmap?

vmap costs ~9.2 us per kernel launch on the RTX 5090 versus ~4.2 us for plain
ops, because BatchedTensor unwrapping happens on the host for every op. A
hand-written batched cell (`bmm` + last-dim layernorm) has no such overhead, and
compiling just the CELL is a tiny graph that builds in seconds -- unlike
compiling the 200x-unrolled step, which is what is currently taking minutes.

This measures the dominant piece (the 200-step unroll, ~73% of the forward)
forward+backward, so we can decide whether the hand-written fallback is worth
building. It does not implement the rollout head.
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/root/inductor_cache')
import torch  # noqa: E402

DEV = torch.device('cuda')
IN, HID, B, T = 152, 500, 16, 200


def cell(xproj_t, hx, Whh, bias):
    x = xproj_t + torch.bmm(hx, Whh)
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True, unbiased=False)
    return torch.relu((x - mean) / (std + 1e-4) + bias)


def unroll(X, Wih, Whh, bias, cell_fn):
    S = X.shape[0]
    hx = torch.zeros(S, B, HID, device=DEV)
    xproj = torch.bmm(X.reshape(S, B * T, IN), Wih).view(S, B, T, HID)
    out = []
    for t in range(T):
        hx = cell_fn(xproj[:, :, t, :], hx, Whh, bias)
        out.append(hx)
    return torch.stack(out, 2)


def timed(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def main():
    tf32 = len(sys.argv) > 1 and sys.argv[1] == 'tf32'
    if tf32:
        torch.set_float32_matmul_precision('high')   # matches train.py::_enable_tf32
    print(f'device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  tf32={tf32}')
    print(f'unroll only (no rollout head), B={B} T={T} H={HID}\n', flush=True)
    print(f"{'S':>4}{'variant':>22}{'ms fwd+bwd':>13}{'ms/model':>11}{'compile s':>11}")

    for S in (1, 8, 26):
        torch.manual_seed(0)
        X = torch.randn(S, B, T, IN, device=DEV)
        Wih = (torch.randn(S, IN, HID, device=DEV) * 0.05).requires_grad_(True)
        Whh = (torch.randn(S, HID, HID, device=DEV) * 0.02).requires_grad_(True)
        bias = torch.zeros(S, 1, HID, device=DEV, requires_grad=True)

        for label, fn, do_compile in (('eager bmm', cell, False),
                                      ('compiled cell', cell, True)):
            try:
                c = torch.compile(fn) if do_compile else fn
                def step():
                    for p in (Wih, Whh, bias):
                        p.grad = None
                    h = unroll(X, Wih, Whh, bias, c)
                    h.square().mean().backward()
                t0 = time.perf_counter(); step(); setup = time.perf_counter() - t0
                med = timed(step)
                print(f'{S:>4}{label:>22}{med*1000:>13.1f}{med*1000/S:>11.2f}{setup:>11.1f}',
                      flush=True)
            except Exception as e:
                print(f'{S:>4}{label:>22}   FAILED {type(e).__name__}: {str(e)[:44]}', flush=True)
            torch.cuda.empty_cache()
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
