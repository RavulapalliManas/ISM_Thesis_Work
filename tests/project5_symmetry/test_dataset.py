import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from project5_symmetry.training.dataset import (
    PackedTrajectoryStore,
    TrajectoryDataset,
)
from utils.data_schema import TRAJECTORY_FMT


def _make_trajectory_npz(path: Path, idx: int, T: int = 10, obs_size: int = 147):
    """Helper to write a valid trajectory .npz file."""
    filepath = path / TRAJECTORY_FMT.format(i=idx)
    np.savez_compressed(
        filepath,
        obs=np.random.randn(T + 1, obs_size).astype(np.float32),
        act_enc=np.random.randn(T, 5).astype(np.float32),
        pos=np.random.randint(0, 18, size=(T + 1, 2)).astype(np.int32),
        heading=np.random.randint(0, 4, size=(T + 1,)).astype(np.int32),
    )
    return filepath


class TestTrajectoryDataset:
    def test_load_single_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _make_trajectory_npz(path, 0, T=10, obs_size=147)
            ds = TrajectoryDataset(str(path))
            assert len(ds) == 1
            obs, act, pos, heading = ds[0]
            assert obs.shape == (11, 147)
            assert act.shape == (10, 5)
            assert pos.shape == (11, 2)
            assert heading.shape == (11,)
            assert obs.dtype == torch.float32
            assert act.dtype == torch.float32

    def test_load_multiple_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for i in range(5):
                _make_trajectory_npz(path, i, T=20, obs_size=147)
            ds = TrajectoryDataset(str(path))
            assert len(ds) == 5

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="No .npz files"):
                TrajectoryDataset(tmp)

    def test_corrupted_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            bad_file = path / TRAJECTORY_FMT.format(i=0)
            bad_file.write_bytes(b"not a valid npz")
            with pytest.raises(Exception):
                TrajectoryDataset(str(path))


class TestPackedTrajectoryStore:
    def test_store_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for i in range(3):
                _make_trajectory_npz(path, i, T=10, obs_size=147)
            store = PackedTrajectoryStore(str(path), device="cpu")
            assert len(store) == 3
            assert store.obs_seq_len == 11
            assert store.act_seq_len == 10
            assert store.obs_size == 147
            assert store.act_size == 5
            assert store.obs_u8.dtype == torch.uint8

    def test_sample_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for i in range(10):
                _make_trajectory_npz(path, i, T=10, obs_size=147)
            store = PackedTrajectoryStore(str(path), device="cpu")
            obs, act = store.sample_batch(batch_size=4)
            assert obs.shape == (4, 11, 147)
            assert act.shape == (4, 10, 5)
            assert obs.dtype == torch.float32
            assert obs.min() >= 0.0 and obs.max() <= 1.0
