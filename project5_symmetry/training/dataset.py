"""
PyTorch Dataset that loads pre-generated trajectory .npz files into RAM.

Each file contains:
  obs     : float32 (T+1, obs_size)  — visual observations [0,1]
  act_enc : float32 (T,   5)         — SpeedHD encoded actions
  pos     : int32   (T+1, 2)         — agent (col, row) positions
  heading : int32   (T+1,)           — agent head directions

For pRNN_th (B=1) training, __getitem__ returns one full trajectory.
"""

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
        files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        self.obs = []
        self.act_enc = []
        self.pos = []
        self.heading = []

        for fname in files:
            d = np.load(os.path.join(data_dir, fname))
            self.obs.append(torch.from_numpy(d['obs']))
            self.act_enc.append(torch.from_numpy(d['act_enc']))
            self.pos.append(torch.from_numpy(d['pos'].astype(np.float32)))
            self.heading.append(torch.from_numpy(d['heading']))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act_enc[idx], self.pos[idx], self.heading[idx]
