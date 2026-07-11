#!/usr/bin/env python3
"""Train ONE pRNN_th on CPU. Does folding require *prediction*?

The GPU ensemble trains the k=5 row of the design. This worker fills in the rest of
the prediction-horizon axis, k in {0,1,3}, which the GPU cannot batch: each k is a
different computation graph, so each would need its own inductor compile and its own
ensemble. On CPU that does not matter -- there are no kernel launches to amortise --
so one model per process is both simpler and measurably faster than vmap
(138 vs 232 ms per 4-thread slot). 36 of these fill the idle EPYC while the GPU works.

The design this completes (arena s2, C2-symmetric):

                k=0        k=1   k=3   k=5
    full HD    (nothing to predict)  ->  odd power > 0, phase decodable
    axis HD    (nothing to predict)  ->  odd power ~ 0, phase at chance

k=0 makes the target `obs[anchor]` -- a pure autoencoder, with no future to predict,
so self-motion is irrelevant to the objective and BOTH encodings should fold. The
HD-invariance effect should therefore *appear only for k > 0*: symmetry is resolved
by prediction, not by observation.

Optimiser and clipping are the reference path: `single_rmsprop` is pinned equal to
train.py::_build_optimizer by test_ensemble.py, and clipping is the stock
`clip_grad_norm_` over one model's parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.hd_encodings import apply_hd  # noqa: E402
from project5_symmetry.experiments.run_ensemble_sweep import (  # noqa: E402
    ANCHOR_SUBSAMPLE_N, CHECKPOINT_STEPS, DROPOUT_P, HIDDEN_INIT_SIGMA,
    HIDDEN_SIZE, NOISE_STD, OBS, ACT, PRED_OFFSET, T, _sample_anchor_idx)
from project5_symmetry.experiments.run_hd_invariance import CONDITIONS, init_seed  # noqa: E402
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402
from utils.Architectures import pRNN_th, prnn_state_dict  # noqa: E402
from utils.thetaRNN import LayerNormRNNCellEager  # noqa: E402

HD_MODES = ('full', 'axis', 'parity', 'const')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True, choices=CONDITIONS)
    ap.add_argument('--hd-mode', required=True, choices=HD_MODES)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', default='/root/runs/horizon')
    ap.add_argument('--data-root', default='/root/data/symmetry')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--log-every', type=int, default=500)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)          # 4 saturates: 8 and 16 buy nothing
    device = torch.device('cpu')
    B, K, A = a.batch_size, a.k, ANCHOR_SUBSAMPLE_N

    # Same init as the GPU run's matching cell, so k is a *paired* comparison.
    torch.manual_seed(init_seed(a.condition, a.hd_mode, a.seed, list(CONDITIONS), HD_MODES))
    model = pRNN_th(obs_size=OBS, act_size=ACT, k=K, hidden_size=HIDDEN_SIZE,
                    cell=LayerNormRNNCellEager, dropp=DROPOUT_P, trunc=T,
                    neuralTimescale=2, predOffset=PRED_OFFSET,
                    hidden_init_sigma=HIDDEN_INIT_SIGMA).to(device)
    model.train()
    opt = ens.single_rmsprop(model, batch_size=B)

    store = PackedTrajectoryStore(str(Path(a.data_root) / a.condition), device=device)
    tag = f'k{K}/{a.condition}/{a.hd_mode}/seed_{a.seed:02d}'
    out = Path(a.out) / f'k{K}' / a.condition / a.hd_mode / f'seed_{a.seed:02d}'
    out.mkdir(parents=True, exist_ok=True)

    meta = {'obs_size': OBS, 'k': K, 'trunc': T, 'hidden_size': HIDDEN_SIZE,
            'batch_size': B, 'noise_std': NOISE_STD, 'dropout_p': DROPOUT_P,
            'hidden_init_sigma': HIDDEN_INIT_SIGMA, 'pred_offset': PRED_OFFSET,
            'n_steps': a.n_steps, 'condition': a.condition, 'hd_mode': a.hd_mode,
            'seed': a.seed, 'runner': 'cpu_worker'}
    log = {'steps': [], 'loss': []}

    print(f'[{tag}] start  threads={a.threads}  steps={a.n_steps}', flush=True)
    t0 = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = store.sample_parallel_batches(1, B)
        obs, act = obs[0], apply_hd(act[0], a.hd_mode)
        state = torch.rand(B, HIDDEN_SIZE) * HIDDEN_INIT_SIGMA
        nm = NOISE_STD * torch.randn(B, T, HIDDEN_SIZE)
        nr = NOISE_STD * torch.randn(B, K, A, HIDDEN_SIZE)
        aidx = _sample_anchor_idx(T - K, device, A)

        pred, _, target = model(obs, act, anchor_idx=aidx, state=state,
                                noise_main=nm, noise_roll=nr)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            log['steps'].append(step)
            log['loss'].append(loss.item())
            if step % 5000 == 0:
                el = time.perf_counter() - t0
                print(f'[{tag}] {step:>6}/{a.n_steps}  loss {loss.item():.5f}  '
                      f'{el/step*1000:.0f} ms/step  eta {(a.n_steps-step)*el/step/3600:.1f} h',
                      flush=True)
        if step in CHECKPOINT_STEPS:
            torch.save({'step': step, 'model': prnn_state_dict(model), 'meta': meta},
                       out / f'ckpt_{step}.pt')

    torch.save({'step': 'final', 'model': prnn_state_dict(model), 'meta': meta},
               out / 'ckpt_final.pt')
    with open(out / 'training_log.json', 'w') as f:
        json.dump(log | {'meta': meta}, f)
    print(f'[{tag}] DONE in {(time.perf_counter()-t0)/3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
