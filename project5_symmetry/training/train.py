"""
Training loop for project5_symmetry rollout pRNN.

Faithful to Levenstein et al. 2024 (Methods p.16-17):
  - Optimizer: RMSProp, alpha=0.95, eps=1e-8  (NOT Adam)
  - Learning rates scaled by init value
  - B=8 per optimizer step (DataLoader batch_size=8)
  - Checkpoints at steps {5000, 10000, 20000, 40000, 60000, 80000, final}
  - sRSA (Euclidean + CityBlock) logged every LOG_INTERVAL steps

Progress tracking:
  - tqdm bar in terminal  →  loss + sRSA live postfix
  - TensorBoard           →  tensorboard --logdir <out_dir>/tb
"""

import os
import json
import time
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

BATCH_SIZE       = 8         # true B=8: DataLoader returns 8 trajectories per call
LOG_INTERVAL     = 1000      # steps between sRSA evaluations
CHECKPOINT_STEPS = {5000, 10000, 20000, 40000, 60000, 80000}
SUBSAMPLE_N      = 4000      # timesteps subsampled for sRSA
HIDDEN_INIT_SIGMA = 0.1      # U(0, σ) hidden state init per trial (Levenstein p.16)


# ── Optimizer ─────────────────────────────────────────────────────────────────

def _build_optimizer(model: pRNN_th) -> torch.optim.Optimizer:
    """
    Per-parameter LRs scaled by init magnitude (Methods p.17):
      LR(W_rec, W_out) = lambda * sqrt(1/H)
      LR(W_in, W_act)  = lambda * sqrt(1/(obs+act))
      LR(bias, scale)  = lambda_b * lambda
    """
    k_io = 1.0 / model.rnn.cell.weight_ih.shape[1]
    k_h  = 1.0 / model.rnn.cell.weight_hh.shape[0]

    return torch.optim.RMSprop(
        [
            {'params': [model.W],                              'lr': GLOBAL_LR * k_h  ** 0.5},
            {'params': [model.W_in],                          'lr': GLOBAL_LR * k_io ** 0.5},
            {'params': [model.W_out],                         'lr': GLOBAL_LR * k_h  ** 0.5},
            {'params': [model.bias, model.rnn.cell.scale],    'lr': GLOBAL_LR * BIAS_LR_SCALE},
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
            # h shape: (1, T', H) from pRNN_th.forward (theta=0 batched path)
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
    n_steps: int = 80000,
    trunc: int = 200,
    batch_size: int = BATCH_SIZE,
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
    n_steps    : total optimizer steps  (each step processes one B=8 batch)
    batch_size : trajectories per optimizer step (default 8)
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
    sample_obs, sample_act, _, _ = dataset[0]
    act_size = sample_act.shape[-1]
    obs_seq_len = sample_obs.shape[0]
    act_seq_len = sample_act.shape[0]
    pin_memory = (device_str == 'cuda')
    # Dataset is RAM-cached, so a few workers are enough to stage batches.
    num_workers = min(4, os.cpu_count() or 2)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,   # B=8: one call → 8 trajectories → one RNN forward
        shuffle=True,
        drop_last=True,          # ensure every batch is exactly B=8
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = pRNN_th(
        obs_size=obs_size,
        act_size=act_size,
        k=k,
        hidden_size=HIDDEN_SIZE,
        cell=LayerNormRNNCell,
        dropp=DROPOUT_P,
        neuralTimescale=NEURAL_TIMESCALE,
        trunc=trunc,
        predOffset=1,
        hidden_init_sigma=HIDDEN_INIT_SIGMA,
    ).to(device)

    # Static buffers for CUDA Graph replay — shapes must stay identical.
    if device.type == 'cuda':
        obs_static = torch.zeros(batch_size, obs_seq_len, obs_size, device=device)
        act_static = torch.zeros(batch_size, act_seq_len, act_size, device=device)
    else:
        obs_static = act_static = None

    # Change 2: compile only the RNN cell, not the whole model.
    # The cell (LayerNormRNNCell) is pure PyTorch matmuls + F.layer_norm + ReLU —
    # Dynamo traces it cleanly. The outer thetaRNNLayer Python loop and pRNN_th's
    # restructure_inputs() (which uses numpy Toeplitz) cannot be compiled without
    # graph breaks, so we leave them as-is.
    # Bug 2 fix: build optimizer BEFORE torch.compile so parameter references
    # are clean (model.rnn.cell.weight_ih etc. live on the original module,
    # not the OptimizedModule wrapper).
    optimizer = _build_optimizer(model)

    compiled = False
    try:
        model.rnn.cell = torch.compile(model.rnn.cell)
        compiled = True
        tqdm.write(f'  [compile] torch.compile(model.rnn.cell) accepted — '
                   f'torch {torch.__version__}  '
                   f'(first cell call triggers JIT; outer loop stays in Python)')
    except Exception as e:
        tqdm.write(f'  [compile] torch.compile(cell) SKIPPED: {e}')
        compiled = False

    use_cuda_graph = (device.type == 'cuda')
    cuda_graph = None
    loss_static = None
    if use_cuda_graph:
        try:
            tqdm.write('  [cuda graph] warming up...')
            warmup_stream = torch.cuda.Stream()
            with torch.cuda.stream(warmup_stream):
                for _ in range(11):
                    pred_w, _, target_w = model(obs_static, act_static)
                    loss_w = F.mse_loss(pred_w, target_w)
                    optimizer.zero_grad(set_to_none=True)
                    loss_w.backward()
                    optimizer.step()
            torch.cuda.current_stream().wait_stream(warmup_stream)

            tqdm.write('  [cuda graph] capturing...')
            cuda_graph = torch.cuda.CUDAGraph()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(cuda_graph):
                pred_static, _, target_static = model(obs_static, act_static)
                loss_static = F.mse_loss(pred_static, target_static)
                loss_static.backward()
            # Keep allocated grad buffers for replay; just zero contents.
            optimizer.zero_grad(set_to_none=False)
            tqdm.write('  [cuda graph] capture complete')
        except Exception as e:
            tqdm.write(f'  [cuda graph] capture failed, falling back: {e}')
            use_cuda_graph = False
            cuda_graph = None
            loss_static = None

    # ── Logging setup ─────────────────────────────────────────────────────────
    tb_dir = os.path.join(out_dir, 'tb')
    writer = SummaryWriter(log_dir=tb_dir, comment=f'_{run_label}' if run_label else '')

    log_dict  = {'steps': [], 'srsa_euclid': [], 'srsa_city': [], 'loss': []}
    ckpt_paths = []
    step       = 0

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

        obs_b, act_b, _, _ = batch   # (B, T+1, obs_size), (B, T, 5)
        if use_cuda_graph and cuda_graph is not None:
            obs_static.copy_(obs_b.to(device, non_blocking=True))
            act_static.copy_(act_b.to(device, non_blocking=True))
            cuda_graph.replay()
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)
            step += 1
            if step % LOG_INTERVAL == 0 or step in CHECKPOINT_STEPS:
                step_loss = loss_static.item()
            else:
                step_loss = float('nan')
        else:
            obs_b = obs_b.to(device, non_blocking=pin_memory)
            act_b = act_b.to(device, non_blocking=pin_memory)

            pred, _, target = model(obs_b, act_b)

            if step == 0 and not use_cuda_graph:
                print("=== CHECK 2: Target Variance ===")
                print(f"obs_target shape: {target.shape}")
                print(f"obs_target mean: {target.mean().item():.6f}")
                print(f"obs_target std: {target.std().item():.6f}")
                print(f"obs_target min: {target.min().item():.6f}")
                print(f"obs_target max: {target.max().item():.6f}")
                print("=== CHECK 3: Prediction Variance ===")
                print(f"pred shape: {pred.shape}")
                print(f"pred mean: {pred.mean().item():.6f}")
                print(f"pred std: {pred.std().item():.6f}")

            loss = F.mse_loss(pred, target)   # B trajectories, one scalar loss

            if step == 0 and not use_cuda_graph:
                print("=== CHECK 5: Loss Integrity ===")
                manual_loss = ((pred - target) ** 2).mean()
                print(f"reported loss: {loss.item():.6f}")
                print(f"manual MSE:    {manual_loss.item():.6f}")
                print(f"loss requires_grad: {loss.requires_grad}")
                print(f"loss grad_fn: {loss.grad_fn}")

            loss.backward()

            if step == 0 and not use_cuda_graph:
                print("=== CHECK 1: Gradient Flow ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
                    else:
                        print(f"{name}: NO GRADIENT")
                print("=== CHECK 4: Learning Rates ===")
                for pg in optimizer.param_groups:
                    print(f"lr={pg['lr']:.2e}  params={[p_name for p_name, p in model.named_parameters() if any(p is pp for pp in pg['params'])]}")

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            step_loss = loss.item()

        pbar.update(1)

        # ── TensorBoard: loss only when materialized (avoid sync every step) ─
        if not np.isnan(step_loss):
            writer.add_scalar('loss/train', step_loss, step)

        if step % LOG_INTERVAL == 0:
            _t0 = time.time()
            h, pos = _collect_hidden_states(model, dataset, SUBSAMPLE_N, device)
            sRSA_e = float(srsa(h, pos, space_metric='euclidean'))
            sRSA_c = float(srsa(h, pos, space_metric='cityblock'))
            dtg    = sRSA_e - sRSA_c
            _srsa_sec = time.time() - _t0
            tqdm.write(f'  [step {step:>6d}] sRSA_e={sRSA_e:.3f}  '
                       f'sRSA_c={sRSA_c:.3f}  ΔTG={dtg:+.3f}  '
                       f'({_srsa_sec:.1f}s)')

            # ── TensorBoard: metrics ──────────────────────────────────────
            writer.add_scalar('metrics/sRSA_euclidean', sRSA_e, step)
            writer.add_scalar('metrics/sRSA_cityblock', sRSA_c, step)
            writer.add_scalar('metrics/delta_TG',       dtg,    step)

            # ── tqdm postfix ──────────────────────────────────────────────
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
