#!/usr/bin/env python3
"""
Resume Phase 0 (P0, seed 0) training from step 40,000 → 80,000.

Run from the repo root:
    python project5_symmetry/training/resume_p0.py

Reads:  project5_symmetry/results/P0/seed_00/ckpt_final.pt
Writes: project5_symmetry/results/P0/seed_00/ckpt_{50,60,70,80}k.pt
        project5_symmetry/results/P0/seed_00/training_log.json  (appended)
        project5_symmetry/results/P0/seed_00/tb/                (new events)
"""

import os
import sys
import json
import time
from pathlib import Path

# Make repo root importable regardless of cwd
_HERE = Path(__file__).resolve().parent          # .../project5_symmetry/training
_ROOT = _HERE.parent.parent                      # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell

from project5_symmetry.training.dataset import TrajectoryDataset
from project5_symmetry.evaluation.metrics import srsa

# Import shared utilities from train.py so we don't duplicate anything
from project5_symmetry.training.train import (
    _build_optimizer,
    _collect_hidden_states,
    HIDDEN_SIZE,
    DROPOUT_P,
    NOISE_STD,
    NEURAL_TIMESCALE,
    GLOBAL_LR,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    WEIGHT_DECAY,
    BATCH_SIZE,
    LOG_INTERVAL,
    OBS_SCALE,
    PRED_OFFSET,
    SUBSAMPLE_N,
    SRSA_EVAL_RUNS,
    HIDDEN_INIT_SIGMA,
    _evaluate_srsa,
)

# ── P0 constants ──────────────────────────────────────────────────────────────
DATA_DIR  = _ROOT / 'project5_symmetry' / 'results' / 'P0' / 'trajectories'
OUT_DIR   = _ROOT / 'project5_symmetry' / 'results' / 'P0' / 'seed_00'
CKPT_IN   = OUT_DIR / 'ckpt_final.pt'
LOG_JSON  = OUT_DIR / 'training_log.json'

OBS_SIZE  = 7 * 7 * 3   # F=7, RGB
ACT_SIZE  = 5
K         = 5
SEED      = 0

START_STEP = 40_000
END_STEP   = 80_000
RESUME_CHECKPOINT_STEPS = {50_000, 60_000, 70_000, 80_000}

LOSS_LO   = 0.03
LOSS_HI   = 0.10


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)
    pin_memory = (device_str == 'cuda')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load existing training log ────────────────────────────────────────────
    with open(LOG_JSON, 'r') as f:
        log_dict = json.load(f)
    # Ensure all expected keys exist (ckpt_paths may not be a list in old logs)
    for key in ('steps', 'srsa_euclid', 'srsa_city', 'loss'):
        if key not in log_dict:
            log_dict[key] = []
    if 'ckpt_paths' not in log_dict or not isinstance(log_dict['ckpt_paths'], list):
        log_dict['ckpt_paths'] = []

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset     = TrajectoryDataset(str(DATA_DIR))
    default_workers = min(8, (os.cpu_count() or 2))
    num_workers = int(os.getenv('PRNN_NUM_WORKERS', default_workers))
    loader      = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = pRNN_th(
        obs_size=OBS_SIZE,
        act_size=ACT_SIZE,
        k=K,
        hidden_size=HIDDEN_SIZE,
        cell=LayerNormRNNCell,
        dropp=DROPOUT_P,
        neuralTimescale=NEURAL_TIMESCALE,
        trunc=200,
        predOffset=PRED_OFFSET,
        hidden_init_sigma=HIDDEN_INIT_SIGMA,
    ).to(device)

    optimizer = _build_optimizer(model, batch_size=BATCH_SIZE)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading checkpoint: {CKPT_IN}')
    ckpt = torch.load(CKPT_IN, map_location=device)

    saved_step = int(ckpt['step'])
    assert saved_step == START_STEP, (
        f'Expected checkpoint at step {START_STEP}, got step {saved_step}. '
        f'Wrong checkpoint file?'
    )

    meta = ckpt.get('meta', {})
    if meta:
        if meta.get('obs_scale') != OBS_SCALE:
            raise RuntimeError(
                f"Checkpoint obs_scale={meta.get('obs_scale')} does not match "
                f'current expected obs_scale={OBS_SCALE}.'
            )
        if meta.get('pred_offset') != PRED_OFFSET:
            raise RuntimeError(
                f"Checkpoint pred_offset={meta.get('pred_offset')} does not match "
                f'current expected pred_offset={PRED_OFFSET}.'
            )

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f'  ✓ checkpoint loaded  (step={saved_step})')

    # ── Compile cell (best-effort, same as train.py) ──────────────────────────
    try:
        model.rnn.cell = torch.compile(model.rnn.cell)
        print('  [compile] torch.compile(model.rnn.cell) accepted')
    except Exception as e:
        print(f'  [compile] SKIPPED: {e}')

    # ── Sanity-check first batch loss before committing to the full run ────────
    model.eval()
    with torch.no_grad():
        sane_batch = next(iter(loader))
        obs_s, act_s = sane_batch[0].to(device), sane_batch[1].to(device)
        pred_s, _, tgt_s = model(obs_s, act_s)
        first_loss = F.mse_loss(pred_s, tgt_s).item()
    model.train()

    print(f'  First-batch loss (sanity): {first_loss:.6f}  '
          f'(expected {LOSS_LO}–{LOSS_HI})')
    if not (LOSS_LO <= first_loss <= LOSS_HI):
        print(f'\nWARNING: first-batch loss {first_loss:.6f} is outside the '
              f'expected range [{LOSS_LO}, {LOSS_HI}].')
        print('This may indicate a bad checkpoint, wrong data, or a mismatch '
              'in hyperparameters. Exiting without training.')
        sys.exit(1)

    # ── TensorBoard (append to existing run directory) ────────────────────────
    tb_dir = OUT_DIR / 'tb'
    writer = SummaryWriter(log_dir=str(tb_dir), comment='_P0_seed0_resume')

    # ── Resume training loop ──────────────────────────────────────────────────
    step        = START_STEP
    n_new_steps = END_STEP - START_STEP
    sRSA_e = sRSA_c = 0.0

    pbar = tqdm(
        total=n_new_steps,
        desc='P0 seed0 resume',
        unit='step',
        dynamic_ncols=True,
        leave=True,
    )

    def _inf_loader():
        while True:
            yield from loader

    for batch in _inf_loader():
        if step >= END_STEP:
            break

        obs_b, act_b, _, _ = batch
        obs_b = obs_b.to(device, non_blocking=pin_memory)
        act_b = act_b.to(device, non_blocking=pin_memory)

        pred, _, target = model(obs_b, act_b)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step     += 1
        step_loss = loss.item()

        pbar.update(1)

        writer.add_scalar('loss/train', step_loss, step)

        if step % LOG_INTERVAL == 0:
            _t0 = time.time()
            sRSA_e, sRSA_c = _evaluate_srsa(
                model, dataset, device, n=SUBSAMPLE_N, repeats=SRSA_EVAL_RUNS
            )
            dtg    = sRSA_e - sRSA_c
            _elapsed = time.time() - _t0
            tqdm.write(f'  [step {step:>6d}] sRSA_e={sRSA_e:.3f}  '
                       f'sRSA_c={sRSA_c:.3f}  ΔTG={dtg:+.3f}  '
                       f'({_elapsed:.1f}s)')

            writer.add_scalar('metrics/sRSA_euclidean', sRSA_e, step)
            writer.add_scalar('metrics/sRSA_cityblock', sRSA_c, step)
            writer.add_scalar('metrics/delta_TG',       dtg,    step)

            pbar.set_postfix({
                'loss':   f'{step_loss:.4f}',
                'sRSA_e': f'{sRSA_e:.3f}',
                'sRSA_c': f'{sRSA_c:.3f}',
                'ΔTG':    f'{dtg:+.3f}',
            })

            log_dict['steps'].append(step)
            log_dict['srsa_euclid'].append(sRSA_e)
            log_dict['srsa_city'].append(sRSA_c)
            log_dict['loss'].append(float(step_loss))

        if step in RESUME_CHECKPOINT_STEPS:
            ckpt_path = OUT_DIR / f'ckpt_{step}.pt'
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'meta': {
                    'obs_scale': OBS_SCALE,
                    'pred_offset': PRED_OFFSET,
                    'dropout_p': DROPOUT_P,
                    'noise_std': NOISE_STD,
                    'hidden_init_sigma': HIDDEN_INIT_SIGMA,
                    'trunc': 200,
                    'srsa_eval_n': SUBSAMPLE_N,
                    'srsa_eval_runs': SRSA_EVAL_RUNS,
                },
            }, ckpt_path)
            log_dict['ckpt_paths'].append(str(ckpt_path))
            writer.add_text('checkpoint', str(ckpt_path), step)
            tqdm.write(f'  [ckpt] saved -> {ckpt_path}')

    pbar.close()
    writer.close()

    # ── Save updated training log ─────────────────────────────────────────────
    with open(LOG_JSON, 'w') as f:
        json.dump(log_dict, f, indent=2)
    print(f'  training_log.json updated ({LOG_JSON})')

    # ── Summary ───────────────────────────────────────────────────────────────
    # Restrict to the new window only
    new_mask   = [s for s in log_dict['steps'] if s > START_STEP]
    new_e_vals = [log_dict['srsa_euclid'][i]
                  for i, s in enumerate(log_dict['steps']) if s > START_STEP]

    if new_e_vals:
        best_e   = max(new_e_vals)
        best_step = new_mask[new_e_vals.index(best_e)]
    else:
        best_e = sRSA_e
        best_step = END_STEP

    gate_steps = [s for s, e in zip(new_mask, new_e_vals) if e > 0.40]
    gate_str   = (f'YES at step {gate_steps[0]}' if gate_steps else 'NO')

    print()
    print('─' * 50)
    print(f'Step 40k→80k complete.')
    print(f'Max sRSA (euclid) in this window: {best_e:.3f} at step {best_step}')
    print(f'sRSA at 80k: {sRSA_e:.3f}')
    print(f'Gate criterion (>0.40) reached: {gate_str}')
    print('─' * 50)


if __name__ == '__main__':
    main()
