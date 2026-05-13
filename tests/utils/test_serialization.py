"""
Tests for utils/serialization.py save/load utilities.

Covers pickle round-trips, JSON round-trips, numpy handling,
auto-creation of parent directories, and ensure_dir idempotency.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils.serialization import (
    ensure_dir,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
)


class TestSerialization:
    def test_save_load_pickle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.pkl"
            obj = {"a": 1, "b": [2, 3], "c": {"d": 4}}
            result = save_pickle(path, obj)
            assert result == path
            assert path.exists()
            loaded = load_pickle(path)
            assert loaded == obj

    def test_save_load_pickle_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.pkl"
            obj = {"arr": np.arange(10), "val": np.float32(3.14)}
            save_pickle(path, obj)
            loaded = load_pickle(path)
            np.testing.assert_array_equal(loaded["arr"], obj["arr"])

    def test_save_load_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            obj = {"name": "test", "values": [1, 2, 3]}
            result = save_json(path, obj)
            assert result == path
            assert path.exists()
            loaded = load_json(path)
            assert loaded == obj

    def test_ensure_dir_creates(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "a" / "b" / "c"
            assert not new_dir.exists()
            result = ensure_dir(new_dir)
            assert result == new_dir
            assert new_dir.is_dir()

    def test_ensure_dir_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ensure_dir(tmp)
            assert result == Path(tmp)
            assert Path(tmp).is_dir()

    def test_save_pickle_mkdir(self):
        """save_pickle should auto-create parent directories."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "deep" / "data.pkl"
            save_pickle(path, {"key": "value"})
            assert path.exists()
            loaded = load_pickle(path)
            assert loaded == {"key": "value"}

    def test_save_json_numpy_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "numpy.json"
            obj = {
                "array": np.array([1, 2, 3]),
                "scalar": np.float64(4.56),
            }
            save_json(path, obj)
            loaded = load_json(path)
            assert loaded["array"] == {"shape": [3], "dtype": "int64"}
            assert loaded["scalar"] == 4.56
