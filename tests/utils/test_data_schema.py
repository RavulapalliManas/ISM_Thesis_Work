import numpy as np
import pytest

from utils.data_schema import (
    TRAJECTORY_FMT,
    TRAJECTORY_KEYS,
    check_trajectory,
)


class TestTrajectorySchema:
    def test_filename_format(self):
        assert TRAJECTORY_FMT.format(i=0) == "traj_00000.npz"
        assert TRAJECTORY_FMT.format(i=42) == "traj_00042.npz"
        assert TRAJECTORY_FMT.format(i=99999) == "traj_99999.npz"

    def test_expected_keys(self):
        assert TRAJECTORY_KEYS == {"obs", "act_enc", "pos", "heading"}

    def test_check_trajectory_valid(self):
        T = 10
        obs_size = 7 * 7 * 3
        data = {
            "obs": np.zeros((T + 1, obs_size), dtype=np.float32),
            "act_enc": np.zeros((T, 5), dtype=np.float32),
            "pos": np.zeros((T + 1, 2), dtype=np.int32),
            "heading": np.zeros((T + 1,), dtype=np.int32),
        }
        check_trajectory(data, name="valid_traj")

    def test_check_trajectory_missing_key(self):
        data = {
            "obs": np.zeros((11, 147), dtype=np.float32),
            "act_enc": np.zeros((10, 5), dtype=np.float32),
        }
        with pytest.raises(KeyError, match="missing keys"):
            check_trajectory(data)

    def test_check_trajectory_wrong_dtype(self):
        data = {
            "obs": np.zeros((11, 147), dtype=np.float64),
            "act_enc": np.zeros((10, 5), dtype=np.float32),
            "pos": np.zeros((11, 2), dtype=np.int32),
            "heading": np.zeros((11,), dtype=np.int32),
        }
        with pytest.raises(TypeError, match="expected dtype float32"):
            check_trajectory(data)
