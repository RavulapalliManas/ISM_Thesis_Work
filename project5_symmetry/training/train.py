"""
Training loop for project5_symmetry rollout pRNN.

Faithful to Levenstein et al. 2024 (Methods p.16-17):
  - Optimizer: RMSProp, alpha=0.95, eps=1e-8  (NOT Adam)
  - Learning rates scaled by init value (see LearningRate rules in paper)
  - B=1 single-trial updates (pRNN_th requires B=1 due to squeeze(0))
  - Effective B=8 via gradient accumulation (project5 optimisation)
  - Checkpoints at steps {5000, 10000, 20000, 40000, final}
  - sRSA (Euclidean + CityBlock) logged every LOG_INTERVAL steps
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell

from project5_symmetry.training.dataset import TrajectoryDataset
from project5_symmetry.evaluation.metrics import srsa

# Paper hyperparameters (Table 1, p.17)
HIDDEN_SIZE   = 500
DROPOUT_P     = 0.15
NOISE_STD     = 0.03
NEURAL_TIMESCALE = 2

GLOBAL_LR     = 2e-3
BIAS_LR_SCALE = 0.1          # lambda_b in paper
WEIGHT_DECAY  = 3e-3         # lambda_wd
RMSPROP_ALPHA = 0.95
RMSPROP_EPS   = 1e-8

ACCUM_STEPS   = 8            # gradient accumulation → effective B=8
LOG_INTERVAL  = 1000
CHECKPOINT_STEPS = {5000, 10000, 20000, 40000}
SUBSAMPLE_N   = 4000          # timesteps for sRSA computation


def _build_optimizer(model: pRNN_th) -> torch.optim.Optimizer:
    """
    Per-parameter learning rates scaled by init magnitude, as in the paper:
      LR(W_rec, W_out) = lambda * sqrt(k_h)
      LR(W_in, W_act)  = lambda * sqrt(k_io)
      LR(b)            = lambda_b * lambda
    Weight decay applied to all params.
    """
    k_io = 1.0 / (model.rnn.cell.weight_ih.shape[1])  # 1/(obs_size+act_size)
    k_h  = 1.0 / model.rnn.cell.weight_hh.shape[0]    # 1/hidden_size

    lr_hh  = GLOBAL_LR * (k_h  ** 0.5)
    lr_io  = GLOBAL_LR * (k_io ** 0.5)
    lr_out = GLOBAL_LR * (k_h  ** 0.5)
    lr_b   = GLOBAL_LR * BIAS_LR_SCALE

    param_groups = [
        {'params': [model.W],     'lr': lr_hh},
        {'params': [model.W_in],  'lr': lr_io},
        {'params': [model.W_out], 'lr': lr_out},
        {'params': [model.bias],  'lr': lr_b},
    ]
    return torch.optim.RMSprop(
        param_groups,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        weight_decay=WEIGHT_DECAY,
    )


def _collect_hidden_states(model: pRNN_th, dataset: TrajectoryDataset,
                            n: int, device) -> tuple:
    """
    Forward pass (no grad) over random trajectories to collect n hidden states.
    Uses rollout step θ=0 (current-step hidden state).

    Returns (hidden (n, H), pos (n, 2)).
    """
    model.eval()
    all_h, all_pos = [], []
    total = 0
    indices = torch.randperm(len(dataset))

    with torch.no_grad():
        for idx in indices:
            obs, act, pos, _ = dataset[idx]
            # pRNN_th expects (1, T+1, obs_size) and (1, T, 5)
            obs_in = obs.unsqueeze(0).to(device)
            act_in = act.unsqueeze(0).to(device)
            _, h, _ = model(obs_in, act_in)
            # h has shape (k+1, T', H) for rollout — take θ=0
            h0 = h[0].cpu().numpy()              # (T', H)
            p  = pos.numpy()[:h0.shape[0]]        # align length

            take = min(h0.shape[0], n - total)
            all_h.append(h0[:take])
            all_pos.append(p[:take])
            total += take
            if total >= n:
                break

    model.train()
    return np.concatenate(all_h, 0), np.concatenate(all_pos, 0)


def train(
    data_dir: str,
    obs_size: int,
    out_dir: str,
    k: int = 5,
    n_steps: int = 40000,
    accum_steps: int = ACCUM_STEPS,
    seed: int = 0,
    device_str: str = None,
) -> dict:
    """
    Train a rollout pRNN on pre-generated trajectories.

    Parameters
    ----------
    data_dir   : directory with traj_*.npz files
    obs_size   : F*F*3 (e.g. 147 for F=7)
    out_dir    : directory for checkpoints and training_log.json
    k          : rollout steps
    n_steps    : total optimizer steps (each = accum_steps fwd passes)
    accum_steps: gradient accumulation window → effective batch size
    seed       : RNG seed

    Returns
    -------
    dict with 'srsa_euclid', 'srsa_city', 'steps', 'loss', 'ckpt_paths'
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    dataset = TrajectoryDataset(data_dir)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                         num_workers=2, pin_memory=(device_str == 'cuda'))

    model = pRNN_th(
        obs_size=obs_size,
        act_size=5,
        k=k,
        hidden_size=HIDDEN_SIZE,
        cell=LayerNormRNNCell,
        dropp=DROPOUT_P,
        neuralTimescale=NEURAL_TIMESCALE,
    ).to(device)

    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = _build_optimizer(model)

    log = {'steps': [], 'srsa_euclid': [], 'srsa_city': [], 'loss': []}
    ckpt_paths = []
    step = 0
    accum_loss = 0.0
    micro = 0

    def _inf_loader():
        while True:
            yield from loader

    for batch in _inf_loader():
        if step >= n_steps:
            break

        obs_b, act_b, _, _ = batch          # (1, T+1, obs_size), (1, T, 5)
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)

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

            if step % LOG_INTERVAL == 0:
                h, pos = _collect_hidden_states(model, dataset, SUBSAMPLE_N, device)
                sRSA_e = srsa(h, pos, space_metric='euclidean')
                sRSA_c = srsa(h, pos, space_metric='cityblock')
                log['steps'].append(step)
                log['srsa_euclid'].append(float(sRSA_e))
                log['srsa_city'].append(float(sRSA_c))
                log['loss'].append(float(accum_loss))
                print(f"step {step:6d} | loss {accum_loss:.4f} | "
                      f"sRSA_e {sRSA_e:.3f} | sRSA_c {sRSA_c:.3f}")
                accum_loss = 0.0

            if step in CHECKPOINT_STEPS:
                ckpt = os.path.join(out_dir, f'ckpt_{step}.pt')
                torch.save({'step': step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, ckpt)
                ckpt_paths.append(ckpt)

    ckpt = os.path.join(out_dir, 'ckpt_final.pt')
    torch.save({'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, ckpt)
    ckpt_paths.append(ckpt)

    log['ckpt_paths'] = ckpt_paths
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    return log
