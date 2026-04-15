import os
import numpy as np
import torch
from torch.utils.data import Dataset

FORWARD_IDX = 2  # action index for forward movement


def _encode_action(act: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Convert integer actions + headings to clean 5-dim SpeedHD encoding.

    Output shape: (T, 5)  — [speed, hd_0, hd_1, hd_2, hd_3]
    speed = 1 iff action == FORWARD, else 0
    hd_i  = one-hot of heading in {0,1,2,3}
    """
    T = len(act)
    out = np.zeros((T, 5), dtype=np.float32)
    out[:, 0] = (act == FORWARD_IDX).astype(np.float32)
    for i in range(4):
        out[:, 1 + i] = (heading == i).astype(np.float32)
    return out


class TrajectoryDataset(Dataset):
    """
    Loads all trajectory .npz files from data_dir into RAM at construction.

    Each item is one trajectory:
        obs     : float32 tensor (T, obs_size)
        act_enc : float32 tensor (T, 5)         — SpeedHD encoding
        pos     : float32 tensor (T, 2)
        heading : int32   tensor (T,)
    """

    def __init__(self, data_dir: str):
        files = sorted(
            f for f in os.listdir(data_dir) if f.endswith('.npz')
        )
        if not files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        self.obs = []
        self.act_enc = []
        self.pos = []
        self.heading = []

        for fname in files:
            d = np.load(os.path.join(data_dir, fname))
            self.obs.append(torch.from_numpy(d['obs']))
            self.act_enc.append(torch.from_numpy(_encode_action(d['act'], d['heading'])))
            self.pos.append(torch.from_numpy(d['pos'].astype(np.float32)))
            self.heading.append(torch.from_numpy(d['heading']))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act_enc[idx], self.pos[idx], self.heading[idx]
