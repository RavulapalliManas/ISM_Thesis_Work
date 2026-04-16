"""
Offline trajectory generation for project5_symmetry.

Action probabilities match the paper (Methods p.14):
  forward=0.60, left=0.15, right=0.15, stop=0.10, others=0.0

Why no RandomActionAgent from utils/agent.py?
  utils/agent.getObservations() accesses env.agent_pos and env.agent_dir
  directly on the wrapper. gymnasium.Wrapper has no __getattr__ forwarding,
  so those attributes (which live on MiniGridEnv) are unreachable from the
  wrapper. We use env.unwrapped.agent_pos instead.
"""

import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from project5_symmetry.environments.arena import make_symmetry_env, PAPER_ACTION_PROBS

FORWARD_IDX = 2   # MiniGrid action index for MoveForward


def _encode_speed_hd(act: np.ndarray, agent_dir: np.ndarray) -> np.ndarray:
    """
    SpeedHD 5-dim encoding faithful to the paper (Methods p.14, form 3):
      [speed, hd_0, hd_1, hd_2, hd_3]
      speed = 1 if action == FORWARD else 0
      hd_i  = one-hot of heading at that timestep in {0,1,2,3}
    Shape: (T, 5)
    """
    T = len(act)
    enc = np.zeros((T, 5), dtype=np.float32)
    enc[:, 0] = (act == FORWARD_IDX).astype(np.float32)
    for i in range(4):
        enc[:, 1 + i] = (agent_dir[:T] == i).astype(np.float32)
    return enc


def collect_trajectory(wrapped_env, T: int, rng: np.random.Generator = None) -> dict:
    """
    Collect one trajectory of T steps using the paper's random-walk policy.

    Accesses agent state via wrapped_env.unwrapped (the raw MiniGridEnv),
    avoiding the gymnasium Wrapper __getattr__ limitation.

    Returns dict:
        obs     : float32 (T+1, obs_size)  — pixel observations, [0,1]
        act_enc : float32 (T,   5)         — SpeedHD encoded actions
        pos     : int32   (T+1, 2)         — agent (col, row) in MiniGrid coords
        heading : int32   (T+1,)           — agent head directions {0,1,2,3}
    """
    if rng is None:
        rng = np.random.default_rng()

    inner = wrapped_env.unwrapped  # SymmetryArena — has agent_pos, agent_dir
    F = inner.agent_view_size
    obs_size = F * F * 3
    n_actions = wrapped_env.action_space.n

    # Sample action sequence with paper probabilities
    probs = np.array(PAPER_ACTION_PROBS[:n_actions], dtype=np.float64)
    probs /= probs.sum()  # normalise (handles floating-point drift)
    actions = rng.choice(n_actions, size=T, p=probs)

    obs_arr  = np.empty((T + 1, obs_size), dtype=np.float32)
    pos_arr  = np.empty((T + 1, 2),        dtype=np.int32)
    dir_arr  = np.empty((T + 1,),          dtype=np.int32)

    raw_obs, _ = wrapped_env.reset()
    obs_arr[0]  = raw_obs['image'].reshape(-1).astype(np.float32) / 255.0
    pos_arr[0]  = inner.agent_pos
    dir_arr[0]  = inner.agent_dir

    for t, a in enumerate(actions):
        raw_obs, _, terminated, truncated, _ = wrapped_env.step(int(a))
        obs_arr[t + 1]  = raw_obs['image'].reshape(-1).astype(np.float32) / 255.0
        pos_arr[t + 1]  = inner.agent_pos
        dir_arr[t + 1]  = inner.agent_dir

    act_enc = _encode_speed_hd(actions, dir_arr)   # heading at t, action at t

    return {
        'obs':     obs_arr,    # (T+1, F*F*3)
        'act_enc': act_enc,    # (T,   5)
        'pos':     pos_arr,    # (T+1, 2)
        'heading': dir_arr,    # (T+1,)
    }


# ------------------------------------------------------------------
# Multiprocessing worker
# ------------------------------------------------------------------

def _worker(args) -> int:
    """Generate a batch of trajectories, save to disk. Returns count written."""
    indices, arena_kwargs, T, out_dir = args
    rng = np.random.default_rng(seed=indices[0])
    env = make_symmetry_env(**arena_kwargs)
    for i in indices:
        traj = collect_trajectory(env, T, rng=rng)
        np.savez_compressed(
            os.path.join(out_dir, f'traj_{i:05d}.npz'),
            obs=traj['obs'],
            act_enc=traj['act_enc'],
            pos=traj['pos'],
            heading=traj['heading'],
        )
    return len(indices)


def generate_dataset(
    wrapped_env,
    n_traj: int,
    T: int,
    out_dir: str,
    n_workers: int = 12,
    desc: str = 'Trajectories',
):
    """
    Generate n_traj trajectories of T steps and save to out_dir.

    Skips files that already exist (resumable).
    Each file: traj_{i:05d}.npz with keys obs, act_enc, pos, heading.
    Shows a tqdm progress bar over completed files.
    """
    os.makedirs(out_dir, exist_ok=True)

    pending = [
        i for i in range(n_traj)
        if not os.path.exists(os.path.join(out_dir, f'traj_{i:05d}.npz'))
    ]
    already_done = n_traj - len(pending)
    if not pending:
        tqdm.write(f'  {desc}: all {n_traj} trajectories already exist, skipping.')
        return

    inner = wrapped_env.unwrapped
    arena_kwargs = {
        'shape': inner.arena_shape,
        'size':  inner.arena_size,
        'U':     inner.U,
        'F':     inner.agent_view_size,
        'seed':  inner._landmark_seed,
    }

    n_workers = min(n_workers, len(pending), cpu_count())
    chunks   = [pending[i::n_workers] for i in range(n_workers)]
    job_args = [(chunk, arena_kwargs, T, out_dir) for chunk in chunks if chunk]

    # tqdm tracks chunks completing; total shows individual files
    with tqdm(total=n_traj, initial=already_done,
              desc=desc, unit='traj', dynamic_ncols=True, leave=False) as pbar:
        with Pool(n_workers) as pool:
            for n_written in pool.imap_unordered(_worker, job_args):
                pbar.update(n_written)
