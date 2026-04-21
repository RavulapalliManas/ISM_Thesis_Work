"""
PyTorch Dataset that preloads pre-generated trajectory .npz files into RAM.

Each file contains:
  obs     : float32 (T+1, obs_size)  — visual observations [0,1]
  act_enc : float32 (T,   5)         — SpeedHD encoded actions
  pos     : int32   (T+1, 2)         — agent (col, row) positions
  heading : int32   (T+1,)           — agent head directions

For pRNN_th (B=1) training, __getitem__ returns one full trajectory.
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads all trajectory .npz files from data_dir into RAM at construction.

    Returns (obs, act_enc, pos, heading) float32 tensors.
    obs and pos are length T+1; act_enc and heading are length T.
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, 'traj_*.npz')))
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        # Preload all trajectories once so __getitem__ is pure RAM access.
        self.cache = [self._load(path) for path in self.files]

    def _load(self, path: str):
        d = np.load(path)
        return (
            torch.from_numpy(d['obs']),
            torch.from_numpy(d['act_enc']),
            torch.from_numpy(d['pos'].astype(np.float32)),
            torch.from_numpy(d['heading']),
        )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]
