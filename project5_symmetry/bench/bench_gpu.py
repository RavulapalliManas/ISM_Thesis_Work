#!/usr/bin/env python3
"""GPU benchmark matrix for the pRNN training step.

Answers, with measurements rather than assertion:
  * what the production config (jit.ScriptModule cell + torch.compile on the cell)
    actually cost, given Dynamo captures zero graphs through TorchScript;
  * what the eager cell + inductor / CUDA graphs buys;
  * where the batch saturates (the step is launch-bound at B=16, so most of the
    GPU is idle);
  * whether batching S seeds into one bmm is really ~free.

Run on ONE gpu:  CUDA_VISIBLE_DEVICES=0 python bench_gpu.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.Architectures import pRNN_th  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager, LayerNormRNNCellScript  # noqa: E402

OBS, ACT, HID, K, A = 147, 5, 500, 5, 32
T = 200
DEV = torch.device('cuda')


def make(cell_cls, batch, compile_what=None, backend='inductor'):
    torch.manual_seed(0)
    m = pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HID, cell=cell_cls,
                dropp=0.15, trunc=T, neuralTimescale=2, predOffset=0,
                hidden_init_sigma=0.1).to(DEV).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    obs = torch.randn(batch, T + 1, OBS, device=DEV)
    act = torch.randn(batch, T, ACT, device=DEV)
    aidx = torch.sort(torch.randperm(T - K, device=DEV)[:A]).values

    with torch.no_grad():          # pre-warm the cached Toeplitz buffers
        m(obs, act, anchor_idx=aidx)

    kw = dict(backend=backend) if backend != 'inductor' else dict(mode='reduce-overhead')
    if compile_what == 'cell':
        m.rnn.cell = torch.compile(m.rnn.cell, **kw)
    elif compile_what == 'model':
        m = torch.compile(m, **kw)

    def step():
        opt.zero_grad(set_to_none=True)
        y, _, tgt = m(obs, act, anchor_idx=aidx)
        ((y - tgt) ** 2).mean().backward()
        opt.step()
    return step


def timed(step, iters=8, warmup=3):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def section_matrix():
    cfgs = [
        ('script cell, no compile (old baseline)',   LayerNormRNNCellScript, None,    'inductor'),
        ('script cell, compile CELL (PRODUCTION)',   LayerNormRNNCellScript, 'cell',  'inductor'),
        ('eager  cell, no compile',                  LayerNormRNNCellEager,  None,    'inductor'),
        ('eager  cell, compile CELL',                LayerNormRNNCellEager,  'cell',  'inductor'),
        ('eager  cell, compile MODEL (inductor+cg)', LayerNormRNNCellEager,  'model', 'inductor'),
        ('eager  cell, MODEL backend=cudagraphs',    LayerNormRNNCellEager,  'model', 'cudagraphs'),
    ]
    print(f"{'config':44}{'ms/step':>10}{'min':>9}{'h/seed@80k':>13}{'h/seed@30k':>13}")
    res = {}
    for name, cls, cw, be in cfgs:
        try:
            t0 = time.perf_counter()
            med, lo, _ = timed(make(cls, 16, cw, be))
            res[name] = med
            print(f'{name:44}{med*1000:>10.1f}{lo*1000:>9.1f}'
                  f'{80000*med/3600:>13.2f}{30000*med/3600:>13.2f}'
                  f'   (setup {time.perf_counter()-t0:.0f}s)', flush=True)
        except Exception as e:
            print(f'{name:44}  FAILED: {type(e).__name__}: {str(e)[:50]}', flush=True)
        finally:
            torch.cuda.empty_cache()
    return res


def section_batch(best_cfg):
    cls, cw, be = best_cfg
    print(f"\n{'B':>6}{'ms/step':>10}{'ms/sample':>12}{'samples/s':>12}{'rel throughput':>16}")
    base = None
    for b in (16, 32, 64, 128, 256, 512, 1024):
        try:
            med, _, _ = timed(make(cls, b, cw, be), iters=5, warmup=2)
            thr = b / med
            base = base or thr
            print(f'{b:>6}{med*1000:>10.1f}{med*1000/b:>12.3f}{thr:>12.1f}{thr/base:>15.2f}x',
                  flush=True)
        except RuntimeError as e:
            print(f'{b:>6}  stop: {str(e)[:48]}', flush=True)
            break
        finally:
            torch.cuda.empty_cache()


def section_seed_batching():
    """Is training S seeds together via bmm really ~free? (unroll only, fwd)"""
    IN = OBS + ACT
    B = 16
    print(f"\n{'S seeds':>9}{'sequential ms':>15}{'bmm ms':>10}{'speedup':>10}{'max|Δ|':>12}")
    for S in (1, 2, 4, 8, 16):
        torch.manual_seed(0)
        Wih = torch.randn(S, IN, HID, device=DEV) * 0.01
        Whh = torch.randn(S, HID, HID, device=DEV) * 0.01
        bias = torch.zeros(S, 1, HID, device=DEV)
        X = torch.randn(S, B, T, IN, device=DEV)

        @torch.no_grad()
        def seq():
            outs = []
            for s in range(S):
                hx = torch.zeros(B, HID, device=DEV)
                xp = (X[s].reshape(B * T, IN) @ Wih[s]).view(B, T, HID)
                for t in range(T):
                    x = xp[:, t, :] + hx @ Whh[s]
                    m_ = x.mean(-1, keepdim=True)
                    sd = x.std(-1, keepdim=True, unbiased=False)
                    hx = torch.relu((x - m_) / (sd + 1e-4) + bias[s])
                outs.append(hx)
            return torch.stack(outs)

        @torch.no_grad()
        def bmm():
            hx = torch.zeros(S, B, HID, device=DEV)
            xp = torch.bmm(X.reshape(S, B * T, IN), Wih).view(S, B, T, HID)
            for t in range(T):
                x = xp[:, :, t, :] + torch.bmm(hx, Whh)
                m_ = x.mean(-1, keepdim=True)
                sd = x.std(-1, keepdim=True, unbiased=False)
                hx = torch.relu((x - m_) / (sd + 1e-4) + bias)
            return hx

        def t(fn, n=4):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts)

        a, b = seq(), bmm()
        ts_, tb_ = t(seq), t(bmm)
        print(f'{S:>9}{ts_*1000:>15.1f}{tb_*1000:>10.1f}{ts_/tb_:>9.2f}x'
              f'{(a-b).abs().max().item():>12.1e}', flush=True)
        torch.cuda.empty_cache()


def main():
    print(f'device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}')
    print(f'shape : B=16 T={T} hidden={HID} k={K} anchors={A}\n', flush=True)

    print('=' * 92 + '\n1. CONFIG MATRIX\n' + '=' * 92, flush=True)
    res = section_matrix()
    prod = res.get('script cell, compile CELL (PRODUCTION)')
    if prod:
        for k in sorted(res, key=lambda k: res[k]):
            if k.startswith('eager'):
                print(f'\nbest eager config: {k}  -> {prod/res[k]:.2f}x vs production', flush=True)
                break

    # pick the fastest eager config that actually built
    order = [('eager  cell, compile MODEL (inductor+cg)', (LayerNormRNNCellEager, 'model', 'inductor')),
             ('eager  cell, MODEL backend=cudagraphs',    (LayerNormRNNCellEager, 'model', 'cudagraphs')),
             ('eager  cell, compile CELL',                (LayerNormRNNCellEager, 'cell',  'inductor')),
             ('eager  cell, no compile',                  (LayerNormRNNCellEager, None,    'inductor'))]
    avail = [(n, c) for n, c in order if n in res]
    best_name, best_cfg = min(avail, key=lambda kv: res[kv[0]])

    print('\n' + '=' * 92 + f'\n2. BATCH SCALING on: {best_name}\n' + '=' * 92, flush=True)
    section_batch(best_cfg)

    print('\n' + '=' * 92 + '\n3. SEED BATCHING (bmm) - is training S seeds together free?\n' + '=' * 92,
          flush=True)
    section_seed_batching()
    print('\ndone', flush=True)


if __name__ == '__main__':
    main()
