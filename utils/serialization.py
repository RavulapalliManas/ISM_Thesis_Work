"""
Pathlib-based serialization utilities for all ISM thesis projects.

Provides save/load for pickle and JSON with automatic parent-directory
creation, numpy-safe JSON encoding, and a unified ensure_dir helper.

Usage:
    from utils.serialization import save_pickle, load_pickle, save_json, load_json

    save_pickle(Path("outputs/data.pkl"), {"key": [1, 2, 3]})
    data = load_json("configs/experiment.json")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: PathLike, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: PathLike) -> Any:
    path = Path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: PathLike, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, indent=indent)
    return path


def load_json(path: PathLike) -> Any:
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
