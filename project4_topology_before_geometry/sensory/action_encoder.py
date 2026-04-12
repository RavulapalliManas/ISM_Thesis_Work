"""Action encoders matching the paper's speed and head-direction inputs."""

from __future__ import annotations

import numpy as np


class ActionEncoder:
    """Encode MiniGrid and RatInABox actions into the paper-style action vectors."""

    _MINIGRID_DIR_TO_INDEX = {3: 0, 1: 1, 0: 2, 2: 3}

    def __init__(self, backend: str):
        backend = str(backend).lower()
        if backend not in {"minigrid", "ratinabox"}:
            raise ValueError(f"Unsupported backend `{backend}`.")
        self.backend = backend
        self.act_dim = 5 if backend == "minigrid" else 13

    def encode(self, raw_actions: np.ndarray, headings: np.ndarray) -> np.ndarray:
        """Encode a full action sequence for the selected backend."""
        raw_actions = np.asarray(raw_actions)
        headings = np.asarray(headings)
        if self.backend == "minigrid":
            return self._encode_minigrid(raw_actions, headings)
        return self._encode_ratinabox(raw_actions, headings)

    def _encode_minigrid(self, raw_actions: np.ndarray, headings: np.ndarray) -> np.ndarray:
        encoded = np.zeros((len(raw_actions), 5), dtype=np.float32)
        encoded[:, 0] = (raw_actions == 2).astype(np.float32)
        for idx, direction in enumerate(headings[: len(raw_actions)]):
            hd_index = self._MINIGRID_DIR_TO_INDEX.get(int(direction))
            if hd_index is not None:
                encoded[idx, 1 + hd_index] = 1.0
        return encoded

    def _encode_ratinabox(self, raw_actions: np.ndarray, headings: np.ndarray) -> np.ndarray:
        encoded = np.zeros((len(raw_actions), 13), dtype=np.float32)
        speeds = np.linalg.norm(np.asarray(raw_actions, dtype=np.float32), axis=1)
        encoded[:, 0] = speeds
        headings = np.asarray(headings[: len(raw_actions)], dtype=np.float32)
        if headings.ndim == 2 and headings.shape[1] == 2:
            angles = np.arctan2(headings[:, 1], headings[:, 0])
        else:
            angles = headings.reshape(-1)
        bins = np.floor(((angles + np.pi) / (2 * np.pi)) * 12).astype(int) % 12
        encoded[np.arange(len(bins)), 1 + bins] = 1.0
        return encoded
