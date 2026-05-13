from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.data_schema import EVAL_FILENAMES, H_KEYS, LOG_FILENAMES, SRSa_KEYS


class ResultStore:
    """
    Unified loader for experiment seed directories.
    Handles filename aliasing (evaluation.pkl / eval.pkl),
    key aliasing (srsa / spatial_rsa / sRSA), and graceful
    fallback when files are missing.
    """

    def __init__(self, seed_dir: Path):
        self.seed_dir = Path(seed_dir)

    # ── File discovery ──────────────────────────────────────────────────────

    def find_evaluation(self) -> Optional[Path]:
        for name in EVAL_FILENAMES:
            p = self.seed_dir / name
            if p.exists():
                return p
        return None

    def find_training_log(self) -> Optional[Path]:
        for name in LOG_FILENAMES:
            p = self.seed_dir / name
            if p.exists():
                return p
        return None

    # ── Loaders ──────────────────────────────────────────────────────────────

    def load_evaluation(self) -> Optional[Dict[str, Any]]:
        path = self.find_evaluation()
        if path is None:
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_training_log(self) -> Optional[Dict[str, Any]]:
        path = self.find_training_log()
        if path is None:
            return None
        with open(path, 'r') as f:
            return json.load(f)

    # ──── Extractors ─────────────────────────────────────────────────────────

    def extract_metric(self, ev: dict, keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in ev:
                val = ev[key]
                if isinstance(val, (list, np.ndarray)):
                    val = float(np.mean(val))
                else:
                    val = float(val)
                if 0.05 <= val <= 0.98:
                    return val
        return None

    def extract_srsa(self, ev: dict) -> Optional[float]:
        return self.extract_metric(ev, SRSa_KEYS)

    def extract_H(self, ev: dict) -> Optional[np.ndarray]:
        for key in H_KEYS:
            if key in ev:
                return np.array(ev[key])
        return None

    # ──── Convenience ────────────────────────────────────────────────────────

    def get_srsa(self) -> Optional[float]:
        ev = self.load_evaluation()
        if ev is None:
            return None
        return self.extract_srsa(ev)

    def get_H(self) -> Optional[np.ndarray]:
        ev = self.load_evaluation()
        if ev is None:
            return None
        return self.extract_H(ev)
