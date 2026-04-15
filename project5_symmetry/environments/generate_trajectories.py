"""
Offline trajectory generation for project5_symmetry.

Uses utils.agent.RandomActionAgent with the paper's action probabilities
(forward=0.6, left=0.15, right=0.15, stop=0.1 — Methods p.14).

Observations are converted to float32 [0,1] and stored with positions
and head directions for downstream dataset loading.
"""

import os
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.agent import RandomActionAgent
from project5_symmetry.environments.arena import (
    make_symmetry_env, PAPER_ACTION_PROBS,
)

FORWARD_IDX = 2


def _obs_to_array(obs_list: list, F: int) -> np.ndarray:
    """
    Convert a list of T+1 observation dicts (from RandomActionAgent) to
    float32 array of shape (T+1, F*F*3) normalised to [0, 1].
    """
    T1 = len(obs_list)
    obs_size = F * F * 3
    out = np.empty((T1, obs_size), dtype=np.float32)
    for t, obs in enumerate(obs_list):
        img = obs['image']                        # (F, F, 3) uint8
        out[t] = img.reshape(-1).astype(np.float32) / 255.0
    return out


def _encode_speed_hd(act: np.ndarray, agent_dir: np.ndarray) -> np.ndarray:
    """
    SpeedHD 5-dim encoding faithful to the paper (Methods p.14, form 3):
      [speed, hd_0, hd_1, hd_2, hd_3]
    speed = 1 if action == FORWARD else 0
    hd_i  = one-hot of current head direction in {0,1,2,3}
    Shape: (T, 5)
    """
    T = len(act)
    enc = np.zeros((T, 5), dtype=np.float32)
    enc[:, 0] = (act == FORWARD_IDX).astype(np.float32)
    for i in range(4):
        enc[:, 1 + i] = (agent_dir[:T] == i).astype(np.float32)
    return enc


def collect_trajectory(wrapped_env, T: int) -> dict:
    """
    Collect one trajectory of T steps using the paper's random-walk policy.

    Returns dict:
        obs     : float32 (T+1, obs_size)  — raw observation sequence
        act_enc : float32 (T,   5)         — SpeedHD encoded actions
        pos     : int32   (T+1, 2)         — agent (col, row) positions
        heading : int32   (T+1,)           — agent head directions
    """
    F = wrapped_env.env.agent_view_size
    agent = RandomActionAgent(
        wrapped_env.action_space,
        default_action_probability=PAPER_ACTION_PROBS,
    )
    obs_list, act_raw, state, _ = agent.getObservations(wrapped_env, T)

    obs = _obs_to_array(obs_list, F)           # (T+1, obs_size)
    act_enc = _encode_speed_hd(act_raw, state['agent_dir'])  # (T, 5)
    pos = state['agent_pos'].astype(np.int32)  # (T+1, 2)  col/row
    heading = state['agent_dir'].astype(np.int32)           # (T+1,)

    return {'obs': obs, 'act_enc': act_enc, 'pos': pos, 'heading': heading}


# ------------------------------------------------------------------
# Multiprocessing worker
# ------------------------------------------------------------------

def _worker(args) -> None:
    """Generate a batch of trajectories and save them to disk."""
    indices, arena_kwargs, T, out_dir = args
    env = make_symmetry_env(**arena_kwargs)
    env.reset()
    for i in indices:
        traj = collect_trajectory(env, T)
        np.savez_compressed(
            os.path.join(out_dir, f'traj_{i:05d}.npz'),
            obs=traj['obs'],
            act_enc=traj['act_enc'],
            pos=traj['pos'],
            heading=traj['heading'],
        )


def generate_dataset(
    wrapped_env,
    n_traj: int,
    T: int,
    out_dir: str,
    n_workers: int = 12,
):
    """
    Generate n_traj trajectories of T steps and save to out_dir.

    Skips files that already exist (resumable).
    Each file: traj_{i:05d}.npz with keys obs, act_enc, pos, heading.
    """
    os.makedirs(out_dir, exist_ok=True)

    pending = [
        i for i in range(n_traj)
        if not os.path.exists(os.path.join(out_dir, f'traj_{i:05d}.npz'))
    ]
    if not pending:
        return

    inner = wrapped_env.env
    arena_kwargs = {
        'shape': inner.arena_shape,
        'size': inner.arena_size,
        'U': inner.U,
        'F': inner.agent_view_size,
        'seed': inner._landmark_seed,
    }

    n_workers = min(n_workers, len(pending), cpu_count())
    chunks = [pending[i::n_workers] for i in range(n_workers)]
    job_args = [(chunk, arena_kwargs, T, out_dir) for chunk in chunks if chunk]

    with Pool(n_workers) as pool:
        pool.map(_worker, job_args)
