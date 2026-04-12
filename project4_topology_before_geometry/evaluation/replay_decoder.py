"""Replay decoding helpers for sleep-state trajectory analysis."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def fit_position_decoder(hidden_wake: np.ndarray, positions: np.ndarray, alpha: float = 1e-3) -> Ridge:
    """Fit a simple linear decoder from hidden states to 2-D position."""
    hidden_wake = np.asarray(hidden_wake, dtype=np.float32)
    if hidden_wake.ndim > 2:
        hidden_wake = hidden_wake.reshape(-1, hidden_wake.shape[-1])
    positions = np.asarray(positions, dtype=np.float32)
    n = min(len(hidden_wake), len(positions))
    decoder = Ridge(alpha=alpha)
    decoder.fit(hidden_wake[:n], positions[:n])
    return decoder


def decode_replay_trajectory(hidden_sleep: np.ndarray, position_decoder, env):
    """Decode sleep activity into positions and summarize geometry consistency."""
    hidden_sleep = np.asarray(hidden_sleep, dtype=np.float32)
    if hidden_sleep.ndim > 2:
        hidden_sleep = hidden_sleep.reshape(-1, hidden_sleep.shape[-1])
    decoded = np.asarray(position_decoder.predict(hidden_sleep), dtype=np.float32)

    consistent = []
    step_lengths = []
    for idx in range(1, len(decoded)):
        consistent.append(env.is_traversable(decoded[idx - 1]) and env.is_traversable(decoded[idx]))
        step_lengths.append(float(np.linalg.norm(decoded[idx] - decoded[idx - 1])))

    if len(step_lengths) == 0:
        step_lengths = [0.0]

    forward_steps = np.sum(np.diff(decoded[:, 0], prepend=decoded[0, 0]) >= 0)
    reverse_steps = max(len(decoded) - forward_steps, 1)
    return {
        "decoded_positions": decoded,
        "fraction_geometrically_consistent": float(np.mean(consistent)) if consistent else 1.0,
        "path_length_distribution": np.histogram(step_lengths, bins=10),
        "forward_vs_reverse_ratio": float(forward_steps / reverse_steps),
    }
