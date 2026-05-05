from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA


def fit_pca_subspace(hidden_states: np.ndarray, k: int) -> np.ndarray:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    n_components = min(k, hidden_states.shape[0], hidden_states.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(hidden_states)
    return pca.components_.T.astype(np.float32)


def canonical_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> Dict[str, np.ndarray | float]:
    cross_covariance = basis_a.T @ basis_b
    _, singular_values, _ = np.linalg.svd(cross_covariance, full_matrices=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.arccos(singular_values)
    distance = float(np.sqrt(np.sum(angles ** 2)))
    return {
        "cross_covariance": cross_covariance.astype(np.float32),
        "singular_values": singular_values.astype(np.float32),
        "angles": angles.astype(np.float32),
        "grassmann_distance": distance,
    }


def projection_ratio(hidden_states: np.ndarray, basis_factual: np.ndarray, basis_hall: np.ndarray) -> np.ndarray:
    factual_proj = hidden_states @ basis_factual @ basis_factual.T
    hall_proj = hidden_states @ basis_hall @ basis_hall.T
    norm_f = np.linalg.norm(factual_proj, axis=1)
    norm_h = np.linalg.norm(hall_proj, axis=1)
    return norm_f / np.clip(norm_f + norm_h, 1e-12, None)


def signed_velocity(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape[0] < 2:
        return np.zeros_like(phi)
    return np.diff(phi, prepend=phi[0])


def projection_dynamics_statistics(
    hallucinated_ratios: Iterable[np.ndarray],
    factual_ratios: Iterable[np.ndarray],
) -> Dict[str, float]:
    hall_means = np.asarray([signed_velocity(phi).mean() for phi in hallucinated_ratios], dtype=np.float64)
    fact_means = np.asarray([signed_velocity(phi).mean() for phi in factual_ratios], dtype=np.float64)
    n = min(hall_means.shape[0], fact_means.shape[0])
    if n == 0:
        return {"hall_mean_delta": 0.0, "fact_mean_delta": 0.0, "t_stat": 0.0, "p_value": 1.0}
    t_stat, p_value = ttest_rel(hall_means[:n], fact_means[:n])
    return {
        "hall_mean_delta": float(hall_means.mean()),
        "fact_mean_delta": float(fact_means.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }

