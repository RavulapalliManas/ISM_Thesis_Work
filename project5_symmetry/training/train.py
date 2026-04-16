"""
Training loop for project5_symmetry rollout pRNN.

Faithful to Levenstein et al. 2024 (Methods p.16-17):
  - Optimizer: RMSProp, alpha=0.95, eps=1e-8  (NOT Adam)
  - Learning rates scaled by init value
  - B=1 single-trial updates (pRNN_th requires B=1 due to squeeze(0))
  - Effective B=8 via gradient accumulation
  - Checkpoints at steps {5000, 10000, 20000, 40000, final}
  - sRSA (Euclidean + CityBlock) logged every LOG_INTERVAL steps

Progress tracking:
  - tqdm bar in terminal  →  loss + sRSA live postfix
  - TensorBoard           →  tensorboard --logdir <out_dir>/tb
"""

import os
import json
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

# ── Paper hyperparameters (Table 1, p.17) ─────────────────────────────────────
HIDDEN_SIZE      = 500
DROPOUT_P        = 0.15
NOISE_STD        = 0.03
NEURAL_TIMESCALE = 2

GLOBAL_LR        = 2e-3
BIAS_LR_SCALE    = 0.1       # lambda_b
WEIGHT_DECAY     = 3e-3      # lambda_wd
RMSPROP_ALPHA    = 0.95
RMSPROP_EPS      = 1e-8

ACCUM_STEPS      = 8         # gradient accumulation → effective B=8
LOG_INTERVAL     = 1000      # steps between sRSA evaluations
CHECKPOINT_STEPS = {5000, 10000, 20000, 40000}
SUBSAMPLE_N      = 4000      # timesteps subsampled for sRSA


# ── Optimizer ─────────────────────────────────────────────────────────────────

def _build_optimizer(model: pRNN_th) -> torch.optim.Optimizer:
    """
    Per-parameter LRs scaled by init magnitude (Methods p.17):
      LR(W_rec, W_out) = lambda * sqrt(1/H)
      LR(W_in, W_act)  = lambda * sqrt(1/(obs+act))
      LR(bias)         = lambda_b * lambda
    """
    k_io = 1.0 / model.rnn.cell.weight_ih.shape[1]
    k_h  = 1.0 / model.rnn.cell.weight_hh.shape[0]

    return torch.optim.RMSprop(
        [
            {'params': [model.W],     'lr': GLOBAL_LR * k_h  ** 0.5},
            {'params': [model.W_in],  'lr': GLOBAL_LR * k_io ** 0.5},
            {'params': [model.W_out], 'lr': GLOBAL_LR * k_h  ** 0.5},
            {'params': [model.bias],  'lr': GLOBAL_LR * BIAS_LR_SCALE},
        ],
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        weight_decay=WEIGHT_DECAY,
    )


# ── Hidden state collection ───────────────────────────────────────────────────

def _collect_hidden_states(model: pRNN_th, dataset: TrajectoryDataset,
                            n: int, device) -> tuple:
    """
    Forward pass (no grad) over random trajectories, collect n hidden states.
    Uses rollout θ=0 (current timestep).

    Returns (hidden: (n, H), pos: (n, 2)).
    """
    model.eval()
    all_h, all_pos = [], []
    total = 0

    with torch.no_grad():
        for idx in torch.randperm(len(dataset)):
            obs, act, pos, _ = dataset[int(idx)]
            _, h, _ = model(obs.unsqueeze(0).to(device),
                            act.unsqueeze(0).to(device))
            # h shape: (T', H)  — rollout returns 2-D tensor for θ=0 path
            h0 = h.squeeze(0).cpu().numpy() if h.dim() == 3 else h.cpu().numpy()
            p  = pos.numpy()[:h0.shape[0]]
            take = min(h0.shape[0], n - total)
            all_h.append(h0[:take])
            all_pos.append(p[:take])
            total += take
            if total >= n:
                break

    model.train()
    return np.concatenate(all_h, 0), np.concatenate(all_pos, 0)


# ── Main training function ────────────────────────────────────────────────────

def train(
    data_dir: str,
    obs_size: int,
    out_dir: str,
    k: int = 5,
    n_steps: int = 40000,
    accum_steps: int = ACCUM_STEPS,
    seed: int = 0,
    device_str: str = None,
    run_label: str = '',       # shown in tqdm bar and TensorBoard run name
) -> dict:
    """
    Train a rollout pRNN on pre-generated trajectories.

    Parameters
    ----------
    data_dir   : directory with traj_*.npz files
    obs_size   : F*F*3 (e.g. 147 for F=7)
    out_dir    : directory for checkpoints, training_log.json, and tb/ logs
    k          : rollout steps ahead
    n_steps    : total optimizer steps  (each step = accum_steps micro-batches)
    accum_steps: gradient accumulation → effective batch size
    seed       : RNG seed
    run_label  : short string shown in tqdm + TensorBoard (e.g. 'P0 seed0')

    Returns
    -------
    dict with keys: steps, loss, srsa_euclid, srsa_city, ckpt_paths
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = TrajectoryDataset(data_dir)
    pin_memory = (device_str == 'cuda')
    num_workers = min(4, os.cpu_count() or 2)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = pRNN_th(
        obs_size=obs_size,
        act_size=5,
        k=k,
        hidden_size=HIDDEN_SIZE,
        cell=LayerNormRNNCell,
        dropp=DROPOUT_P,
        neuralTimescale=NEURAL_TIMESCALE,
    ).to(device)

    compiled = False
    try:
        model = torch.compile(model)
        compiled = True
    except Exception:
        compiled = False

    optimizer = _build_optimizer(model)

    # ── Logging setup ─────────────────────────────────────────────────────────
    tb_dir = os.path.join(out_dir, 'tb')
    writer = SummaryWriter(log_dir=tb_dir, comment=f'_{run_label}' if run_label else '')

    log_dict  = {'steps': [], 'srsa_euclid': [], 'srsa_city': [], 'loss': []}
    ckpt_paths = []
    step       = 0
    accum_loss = 0.0
    micro      = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    pbar = tqdm(
        total=n_steps,
        desc=run_label or 'train',
        unit='step',
        dynamic_ncols=True,
        leave=True,
    )

    def _inf_loader():
        while True:
            yield from loader

    sRSA_e = sRSA_c = 0.0   # tracked for tqdm postfix

    for batch in _inf_loader():
        if step >= n_steps:
            break

        obs_b, act_b, _, _ = batch       # (1, T+1, obs_size), (1, T, 5)
        obs_b = obs_b.to(device, non_blocking=pin_memory)
        act_b = act_b.to(device, non_blocking=pin_memory)

        pred, _, target = model(obs_b, act_b)
        loss = F.mse_loss(pred, target) / accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro += 1

        if micro == accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro = 0
            pbar.update(1)

            # ── TensorBoard: loss every step ──────────────────────────────────
            writer.add_scalar('loss/train', accum_loss, step)

            if step % LOG_INTERVAL == 0:
                h, pos = _collect_hidden_states(model, dataset, SUBSAMPLE_N, device)
                sRSA_e = float(srsa(h, pos, space_metric='euclidean'))
                sRSA_c = float(srsa(h, pos, space_metric='cityblock'))
                dtg    = sRSA_e - sRSA_c

                # ── TensorBoard: metrics ──────────────────────────────────────
                writer.add_scalar('metrics/sRSA_euclidean', sRSA_e, step)
                writer.add_scalar('metrics/sRSA_cityblock', sRSA_c, step)
                writer.add_scalar('metrics/delta_TG',       dtg,    step)

                # ── tqdm postfix ──────────────────────────────────────────────
                pbar.set_postfix({
                    'loss':   f'{accum_loss:.4f}',
                    'sRSA_e': f'{sRSA_e:.3f}',
                    'sRSA_c': f'{sRSA_c:.3f}',
                    'ΔTG':    f'{dtg:+.3f}',
                })

                log_dict['steps'].append(step)
                log_dict['srsa_euclid'].append(sRSA_e)
                log_dict['srsa_city'].append(sRSA_c)
                log_dict['loss'].append(float(accum_loss))
                accum_loss = 0.0

            if step in CHECKPOINT_STEPS:
                ckpt = os.path.join(out_dir, f'ckpt_{step}.pt')
                torch.save({'step': step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, ckpt)
                ckpt_paths.append(ckpt)
                writer.add_text('checkpoint', ckpt, step)

    pbar.close()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    ckpt = os.path.join(out_dir, 'ckpt_final.pt')
    torch.save({'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, ckpt)
    ckpt_paths.append(ckpt)

    writer.close()

    log_dict['ckpt_paths'] = ckpt_paths
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log_dict, f, indent=2)

    compile_note = " (torch.compile enabled)" if compiled else ""
    tqdm.write(f'  ✓ {run_label or "train"} done{compile_note} — '
               f'sRSA_e={sRSA_e:.3f}  sRSA_c={sRSA_c:.3f}  '
               f'tb → {tb_dir}')

    return log_dict
