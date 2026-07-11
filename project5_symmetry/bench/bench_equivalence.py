#!/usr/bin/env python3
"""Equivalence gate for the pRNN training-path optimisations.

Compares the legacy path (TorchScript cell) against the new path (eager
nn.Module cell) on three levels:

  1. FORWARD      must be bit-exact (max |Δ| == 0).
  2. BACKWARD     must agree to within the legacy path's *own* run-to-run
                  nondeterminism. TorchScript's fused autodiff is not
                  reproducible with itself (~1e-10 on this model), so demanding
                  bit-exactness against it is not a meaningful test. The eager
                  path, by contrast, is exactly reproducible.
  3. TRAINING     N optimiser steps must track to fp32 tolerance.

Usage:
    python project5_symmetry/bench/bench_equivalence.py
    PRNN_HOIST_INPUT_PROJ=1 python project5_symmetry/bench/bench_equivalence.py

Exit status is non-zero if any level fails.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager, LayerNormRNNCellScript  # noqa: E402

OBS, ACT = 147, 5


def build(cell_cls, k, hidden, trunc, dropp=0.0):
    torch.manual_seed(0)
    return pRNN_th(obs_size=OBS, act_size=ACT, k=k, hidden_size=hidden, cell=cell_cls,
                   dropp=dropp, trunc=trunc, neuralTimescale=2, predOffset=0,
                   hidden_init_sigma=0.1)


def forward_once(model, obs, act, aidx, seed=9):
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        return model(obs, act, anchor_idx=aidx)


def grads_once(model, obs, act, aidx, seed=9):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    y, _, tgt = model(obs, act, anchor_idx=aidx)
    ((y - tgt) ** 2).mean().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def max_grad_delta(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


def train_steps(model, batches, aidx, lr, seed0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, hs = [], []
    for i, (obs, act) in enumerate(batches):
        torch.manual_seed(seed0 + i)
        opt.zero_grad(set_to_none=True)
        y, h, tgt = model(obs, act, anchor_idx=aidx)
        loss = ((y - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.detach().clone())
        hs.append(h.detach().clone())
    return torch.stack(losses), torch.stack(hs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--T', type=int, default=40)
    ap.add_argument('--hidden', type=int, default=500)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--anchors', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rtol', type=float, default=1e-4,
                    help='relative tolerance for the N-step training trajectory')
    ap.add_argument('--noise-margin', type=float, default=10.0,
                    help='backward delta may be at most this multiple of the legacy self-noise')
    a = ap.parse_args()

    hoist = os.environ.get('PRNN_HOIST_INPUT_PROJ', '0') == '1'
    print(f'B={a.batch} T={a.T} hidden={a.hidden} k={a.k} anchors={a.anchors} '
          f'steps={a.steps}  hoist_input_proj={hoist}\n')

    torch.manual_seed(1234)
    obs = torch.randn(a.batch, a.T + 1, OBS)
    act = torch.randn(a.batch, a.T, ACT)
    aidx = torch.sort(torch.randperm(a.T - a.k)[:min(a.anchors, a.T - a.k)]).values
    batches = [(torch.randn(a.batch, a.T + 1, OBS), torch.randn(a.batch, a.T, ACT))
               for _ in range(a.steps)]

    ref = build(LayerNormRNNCellScript, a.k, a.hidden, a.T)
    new = build(LayerNormRNNCellEager, a.k, a.hidden, a.T)
    new.load_state_dict(ref.state_dict())
    assert not hasattr(ref.rnn.cell, 'forward_projected')
    assert hasattr(new.rnn.cell, 'forward_projected')

    ok = True

    # ---- 1. forward ------------------------------------------------------
    y1, h1, t1 = forward_once(ref, obs, act, aidx)
    y2, h2, t2 = forward_once(new, obs, act, aidx)
    d_fwd = max((y1 - y2).abs().max().item(),
                (h1 - h2).abs().max().item(),
                (t1 - t2).abs().max().item())
    good = (d_fwd == 0.0)
    ok &= good
    print('1. FORWARD (must be exactly 0)')
    print(f'     max |Δ(y, h, target)| = {d_fwd:.3e}   {"OK" if good else "FAIL"}\n')

    # ---- 2. backward vs legacy self-noise --------------------------------
    # Warm the TorchScript profiling executor first: its first couple of calls run
    # an unoptimised graph, so a cold measurement overstates the self-noise.
    for _ in range(3):
        grads_once(build(LayerNormRNNCellScript, a.k, a.hidden, a.T), obs, act, aidx)
    g_ref_a = grads_once(build(LayerNormRNNCellScript, a.k, a.hidden, a.T), obs, act, aidx)
    g_ref_b = grads_once(build(LayerNormRNNCellScript, a.k, a.hidden, a.T), obs, act, aidx)
    legacy_noise = max_grad_delta(g_ref_a, g_ref_b)

    e_a = build(LayerNormRNNCellEager, a.k, a.hidden, a.T); e_a.load_state_dict(ref.state_dict())
    e_b = build(LayerNormRNNCellEager, a.k, a.hidden, a.T); e_b.load_state_dict(ref.state_dict())
    g_new_a, g_new_b = grads_once(e_a, obs, act, aidx), grads_once(e_b, obs, act, aidx)
    new_noise = max_grad_delta(g_new_a, g_new_b)

    cross = max_grad_delta(g_ref_a, g_new_a)
    budget = max(legacy_noise * a.noise_margin, 1e-12)
    good = cross <= budget
    ok &= good
    print('2. BACKWARD (vs the legacy path\'s own nondeterminism)')
    print(f'     legacy vs legacy (self-noise) = {legacy_noise:.3e}')
    print(f'     new    vs new    (self-noise) = {new_noise:.3e}'
          f'{"   <- exactly reproducible" if new_noise == 0.0 else ""}')
    print(f'     legacy vs new                 = {cross:.3e}   budget {budget:.3e}   '
          f'{"OK" if good else "FAIL"}\n')

    # ---- 3. N-step training trajectory ------------------------------------
    l_ref, h_ref = train_steps(build(LayerNormRNNCellScript, a.k, a.hidden, a.T),
                               batches, aidx, a.lr, seed0=777)
    e = build(LayerNormRNNCellEager, a.k, a.hidden, a.T); e.load_state_dict(ref.state_dict())
    l_new, h_new = train_steps(e, batches, aidx, a.lr, seed0=777)

    print(f'3. TRAINING ({a.steps} Adam steps, relative tolerance {a.rtol:g})')
    print(f"     {'tensor':>8}  {'max |Δ|':>12}  {'relative':>12}  verdict")
    for name, o, n in (('loss', l_ref, l_new), ('hidden', h_ref, h_new)):
        d = (o - n).abs().max().item()
        rel = d / (o.abs().max().item() or 1.0)
        g = rel <= a.rtol
        ok &= g
        print(f'     {name:>8}  {d:12.3e}  {rel:12.3e}  {"OK" if g else "FAIL"}')
    print(f'\n     final loss  legacy={l_ref[-1].item():.8f}  new={l_new[-1].item():.8f}')

    print('\n' + ('PASS - the optimised path matches the legacy path to within its own '
                  'reproducibility.' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
