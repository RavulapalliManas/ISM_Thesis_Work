"""Representational-drift metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_rdm_delta(hidden_states_t1: np.ndarray, hidden_states_t2: np.ndarray, max_points: int = 200, seed: int = 42) -> float:
    """Normalized Frobenius change between two cosine-distance RDMs."""
    hidden_states_t1 = np.asarray(hidden_states_t1, dtype=np.float32)
    hidden_states_t2 = np.asarray(hidden_states_t2, dtype=np.float32)
    n = min(len(hidden_states_t1), len(hidden_states_t2))
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if n > max_points:
        keep = np.sort(rng.choice(n, size=max_points, replace=False))
        hidden_states_t1 = hidden_states_t1[keep]
        hidden_states_t2 = hidden_states_t2[keep]
    else:
        hidden_states_t1 = hidden_states_t1[:n]
        hidden_states_t2 = hidden_states_t2[:n]
    rdm_t1 = squareform(pdist(hidden_states_t1, metric="cosine"))
    rdm_t2 = squareform(pdist(hidden_states_t2, metric="cosine"))
    denominator = np.linalg.norm(rdm_t1, ord="fro")
    if denominator <= 1e-9:
        return 0.0
    return float(np.linalg.norm(rdm_t2 - rdm_t1, ord="fro") / denominator)
