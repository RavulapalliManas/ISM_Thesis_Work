"""
Trajectory data schema constants and validation for the ISM thesis.

Defines the canonical file format for trajectory .npz files shared
across trajectory generation (generate_trajectories.py), dataset
loading (dataset.py), and downstream evaluation.

Schema per .npz file:
    obs     : float32 (T+1, obs_size)  — visual observations [0,1]
    act_enc : float32 (T,   5)         — SpeedHD encoded actions
    pos     : int32   (T+1, 2)         — agent (col, row) positions
    heading : int32   (T+1,)           — agent head directions {0,1,2,3}

Usage:
    from utils.data_schema import check_trajectory, TRAJECTORY_FMT
    check_trajectory(npz_data, name="traj_00000.npz")
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


TRAJECTORY_FMT = "traj_{i:05d}.npz"

TRAJECTORY_KEYS = {"obs", "act_enc", "pos", "heading"}

TRAJECTORY_SHAPES: Dict[str, Tuple[str, ...]] = {
    "obs":     ("T+1", "obs_size"),
    "act_enc": ("T",   "5"),
    "pos":     ("T+1", "2"),
    "heading": ("T+1",),
}

TRAJECTORY_DTYPES: Dict[str, np.dtype] = {
    "obs":     np.dtype(np.float32),
    "act_enc": np.dtype(np.float32),
    "pos":     np.dtype(np.int32),
    "heading": np.dtype(np.int32),
}

EVAL_FILENAMES = ["evaluation.pkl", "eval.pkl"]
LOG_FILENAMES = ["training_log.json", "training_curve.json", "metrics.json"]

SRSa_KEYS = ["srsa", "spatial_rsa", "sRSA"]
H_KEYS = ["position_hidden", "hidden_states", "H"]


def check_trajectory(data: dict, name: str = "trajectory") -> None:
    missing = TRAJECTORY_KEYS - set(data.keys())
    if missing:
        raise KeyError(f"{name}: missing keys {missing}")

    for key in TRAJECTORY_KEYS:
        arr = data[key]
        expected_dtype = TRAJECTORY_DTYPES[key]
        if arr.dtype != expected_dtype:
            raise TypeError(
                f"{name}[{key}]: expected dtype {expected_dtype}, got {arr.dtype}"
            )
