import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

from project5_symmetry.environments.arena import Arena, LEFT, RIGHT, FORWARD, STAY


def random_walk(arena: Arena, T: int, p_forward: float = 0.6, seed: int = 0) -> dict:
    """
    Generate a single trajectory of T steps via random walk.

    Action probs: forward=p_forward, left/right/stay each get (1-p_forward)/3.

    Returns dict with keys:
        obs     : float32 (T, obs_size)
        act     : int32   (T,)
        pos     : int32   (T, 2)   row/col each timestep
        heading : int32   (T,)
    """
    rng = np.random.default_rng(seed)
    p_other = (1.0 - p_forward) / 3.0
    action_probs = [p_other, p_other, p_forward, p_other]  # L, R, F, Stay

    passable = arena.passable_positions
    start_idx = rng.integers(0, len(passable))
    pos = passable[start_idx]
    heading = int(rng.integers(0, 4))

    obs_buf = np.empty((T, arena.obs_size), dtype=np.float32)
    act_buf = np.empty(T, dtype=np.int32)
    pos_buf = np.empty((T, 2), dtype=np.int32)
    head_buf = np.empty(T, dtype=np.int32)

    for t in range(T):
        obs_buf[t] = arena.get_obs(pos, heading)
        action = int(rng.choice(4, p=action_probs))
        act_buf[t] = action
        pos_buf[t] = pos
        head_buf[t] = heading
        pos, heading = arena.step(pos, heading, action)

    return {'obs': obs_buf, 'act': act_buf, 'pos': pos_buf, 'heading': head_buf}


def _worker(args) -> None:
    """Worker that generates a batch of trajectories and saves them."""
    indices, arena_kwargs, T, p_forward, out_dir, base_seed = args
    arena = Arena(**arena_kwargs)
    for i in indices:
        traj = random_walk(arena, T, p_forward, seed=base_seed + i)
        np.savez_compressed(
            os.path.join(out_dir, f'traj_{i:05d}.npz'),
            obs=traj['obs'],
            act=traj['act'],
            pos=traj['pos'],
            heading=traj['heading'],
        )


def generate_dataset(
    arena: Arena,
    n_traj: int,
    T: int,
    out_dir: str,
    n_workers: int = 12,
    p_forward: float = 0.6,
    base_seed: int = 0,
):
    """
    Generate n_traj trajectories of length T in parallel and save to out_dir.

    Files are named traj_00000.npz … traj_{n_traj-1:05d}.npz.
    Skips files that already exist (resumable).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Determine which trajectories still need generating
    pending = [i for i in range(n_traj)
               if not os.path.exists(os.path.join(out_dir, f'traj_{i:05d}.npz'))]
    if not pending:
        return

    arena_kwargs = {
        'shape': arena.shape,
        'size': arena.size,
        'U': arena.U,
        'F': arena.F,
        'seed': arena.seed,
    }

    # Split pending indices across workers
    n_workers = min(n_workers, len(pending), cpu_count())
    chunks = [pending[i::n_workers] for i in range(n_workers)]
    args = [(chunk, arena_kwargs, T, p_forward, out_dir, base_seed) for chunk in chunks]

    with Pool(n_workers) as pool:
        pool.map(_worker, args)
