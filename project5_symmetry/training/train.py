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

CHECKPOINT_STEPS = {5000, 10000, 20000, 40000}
LOG_INTERVAL = 1000
HIDDEN_SIZE = 500
LR = 1e-3
SUBSAMPLE_N = 4000  # timesteps used for sRSA computation

# NOTE: pRNN_th.restructure_inputs uses squeeze(0) which assumes batch_size=1.
# We simulate B=8 via gradient accumulation: accumulate gradients over
# ACCUM_STEPS single-trajectory forward passes before each optimizer.step().
ACCUM_STEPS = 8


def _collect_hidden_states(model: pRNN_th, dataset: TrajectoryDataset, n: int, device):
    """
    Forward pass through n random timesteps (no grad) to collect hidden states.
    Returns (hidden numpy (n,H), pos numpy (n,2)).
    """
    model.eval()
    all_h, all_pos = [], []
    total = 0
    indices = torch.randperm(len(dataset))

    with torch.no_grad():
        for idx in indices:
            obs, act, pos, _ = dataset[idx]
            obs = obs.unsqueeze(0).to(device)   # (1, T, obs_size)
            act = act.unsqueeze(0).to(device)   # (1, T, 5)
            _, h, _ = model(obs, act)            # h: (k+1, T', H) for pRNN_th
            # Use only the first rollout step's hidden states as the 'wake' activity
            if h.dim() == 3:
                h_np = h[0].cpu().numpy()       # (T', H)
            else:
                h_np = h.squeeze(0).cpu().numpy()
            p_np = pos.numpy()[:h_np.shape[0]]  # align length

            remaining = n - total
            take = min(h_np.shape[0], remaining)
            all_h.append(h_np[:take])
            all_pos.append(p_np[:take])
            total += take
            if total >= n:
                break

    model.train()
    return np.concatenate(all_h, axis=0), np.concatenate(all_pos, axis=0)


def train(
    data_dir: str,
    obs_size: int,
    out_dir: str,
    k: int = 5,
    T: int = 200,
    accum_steps: int = ACCUM_STEPS,
    n_steps: int = 40000,
    seed: int = 0,
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> dict:
    """
    Train a rollout pRNN on pre-generated trajectories.

    pRNN_th requires batch_size=1; we use gradient accumulation over
    accum_steps trajectories to approximate effective batch size B=8.

    Parameters
    ----------
    data_dir    : directory with traj_*.npz files
    obs_size    : F*F*3 (e.g. 147 for F=7)
    out_dir     : directory for checkpoints and log JSON
    k           : rollout steps (theta parameter)
    T           : sequence length (informational; actual T from dataset)
    accum_steps : gradient accumulation steps (effective batch size)
    n_steps     : total optimizer steps (each step = accum_steps forward passes)
    seed        : torch/numpy RNG seed

    Returns
    -------
    dict with keys: 'srsa_euclid', 'srsa_city', 'steps', 'ckpt_paths', 'loss'
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    dataset = TrajectoryDataset(data_dir)
    # B=1 loader; accumulate manually
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=2, pin_memory=(device_str == 'cuda'))

    model = pRNN_th(
        obs_size=obs_size,
        act_size=5,
        k=k,
        hidden_size=HIDDEN_SIZE,
        cell=LayerNormRNNCell,
        neuralTimescale=2,
    ).to(device)

    try:
        model = torch.compile(model)
    except Exception:
        pass  # torch.compile may not be available on all platforms

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log = {'steps': [], 'srsa_euclid': [], 'srsa_city': [], 'loss': []}
    ckpt_paths = []
    step = 0
    accum_loss = 0.0
    micro_step = 0

    def _inf_loader():
        while True:
            yield from loader

    for batch in _inf_loader():
        if step >= n_steps:
            break

        obs_b, act_b, pos_b, _ = batch      # (1, T, ...)
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)

        pred, _, target = model(obs_b, act_b)
        loss = F.mse_loss(pred, target) / accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step == accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro_step = 0

            if step % LOG_INTERVAL == 0:
                h_np, pos_np = _collect_hidden_states(model, dataset, SUBSAMPLE_N, device)
                sRSA_e = srsa(h_np, pos_np, neural_metric='cosine', space_metric='euclidean')
                sRSA_c = srsa(h_np, pos_np, neural_metric='cosine', space_metric='cityblock')
                log['steps'].append(step)
                log['srsa_euclid'].append(float(sRSA_e))
                log['srsa_city'].append(float(sRSA_c))
                log['loss'].append(float(accum_loss))
                print(f"step {step:6d} | loss {accum_loss:.4f} | "
                      f"sRSA_e {sRSA_e:.3f} | sRSA_c {sRSA_c:.3f}")
                accum_loss = 0.0

            if step in CHECKPOINT_STEPS:
                ckpt_path = os.path.join(out_dir, f'ckpt_{step}.pt')
                torch.save({'step': step, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, ckpt_path)
                ckpt_paths.append(ckpt_path)

    # Final checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt_final.pt')
    torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, ckpt_path)
    ckpt_paths.append(ckpt_path)

    log['ckpt_paths'] = ckpt_paths
    with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    return log
