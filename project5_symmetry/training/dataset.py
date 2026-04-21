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


class PackedTrajectoryStore:
    """
    Packed trajectory tensors for direct device-side batch sampling.

    Observations are quantized to uint8 in memory and decoded back to float32
    in [0, 1] only when a batch is sampled. This keeps the full offline dataset
    compact enough to reside on the GPU for the fast trainer.
    """

    def __init__(
        self,
        data_dir: str,
        device: torch.device | str,
        act_dtype: torch.dtype = torch.float16,
    ):
        base = TrajectoryDataset(data_dir)
        self.device = torch.device(device)
        self.num_traj = len(base)

        obs = torch.stack([item[0] for item in base.cache], dim=0).clamp_(0.0, 1.0)
        act = torch.stack([item[1] for item in base.cache], dim=0)
        pos = torch.stack([item[2] for item in base.cache], dim=0)
        heading = torch.stack([item[3] for item in base.cache], dim=0)

        self.obs_u8 = torch.round(obs * 255.0).to(torch.uint8).to(self.device)
        self.act = act.to(act_dtype).to(self.device)
        # Positions/headings are kept on CPU for evaluation code that expects numpy.
        self.pos = pos
        self.heading = heading

        self.obs_seq_len = self.obs_u8.shape[1]
        self.act_seq_len = self.act.shape[1]
        self.obs_size = self.obs_u8.shape[2]
        self.act_size = self.act.shape[2]

    def __len__(self):
        return self.num_traj

    def _decode_obs(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return obs_u8.float().mul_(1.0 / 255.0)

    def sample_batch(
        self,
        batch_size: int,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indices is None:
            indices = torch.randint(self.num_traj, (batch_size,), device=self.device)
        obs = self._decode_obs(self.obs_u8[indices])
        act = self.act[indices].float()
        return obs, act

    def sample_parallel_batches(
        self,
        n_groups: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(self.num_traj, (n_groups, batch_size), device=self.device)
        obs = self._decode_obs(self.obs_u8[indices])
        act = self.act[indices].float()
        return obs, act
