from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import spearmanr

from reasoning_geometry.common import sem


def _upper_triangular_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def spearman_rsa(matrix_a: np.ndarray, matrix_b: np.ndarray) -> Tuple[float, float]:
    vec_a = _upper_triangular_values(np.asarray(matrix_a))
    vec_b = _upper_triangular_values(np.asarray(matrix_b))
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0, 1.0
    rho, p_value = spearmanr(vec_a, vec_b)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p_value)


def batched_spearman_rsa(
    matrices_a: Iterable[np.ndarray],
    matrices_b: Iterable[np.ndarray],
) -> Dict[str, np.ndarray | float]:
    rhos: List[float] = []
    p_values: List[float] = []
    for mat_a, mat_b in zip(matrices_a, matrices_b):
        rho, p_value = spearman_rsa(mat_a, mat_b)
        rhos.append(rho)
        p_values.append(p_value)
    return {
        "values": np.asarray(rhos, dtype=np.float32),
        "p_values": np.asarray(p_values, dtype=np.float32),
        "mean": float(np.mean(rhos)) if rhos else 0.0,
        "se": sem(rhos),
    }

