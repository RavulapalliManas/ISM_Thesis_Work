from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from reasoning_geometry.common import pairwise_euclidean


def twonn_intrinsic_dimensionality(hidden_states: np.ndarray) -> float:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    if hidden_states.shape[0] < 3:
        return 0.0
    distances = pairwise_euclidean(hidden_states)
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, kth=1, axis=1)[:, :2]
    nearest.sort(axis=1)
    mu = nearest[:, 1] / np.clip(nearest[:, 0], 1e-12, None)
    logs = np.log(np.clip(mu, 1.0 + 1e-12, None))
    return float(1.0 / np.mean(logs)) if np.mean(logs) > 0 else 0.0


def bootstrap_twonn(hidden_states: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, float]:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    if hidden_states.shape[0] < 3:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    samples: List[float] = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(hidden_states.shape[0], size=hidden_states.shape[0], replace=True)
        sample = hidden_states[indices]
        samples.append(twonn_intrinsic_dimensionality(sample))
    return {
        "estimate": float(np.mean(samples)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def classical_mds(hidden_states: np.ndarray, n_components: int = 2) -> np.ndarray:
    distances = pairwise_euclidean(hidden_states)
    n = distances.shape[0]
    identity = np.eye(n)
    ones = np.ones((n, n)) / n
    centered = -0.5 * (identity - ones) @ (distances ** 2) @ (identity - ones)
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order][:n_components], 0.0)
    eigenvectors = eigenvectors[:, order][:, :n_components]
    return eigenvectors * np.sqrt(eigenvalues)


def mds_stress(hidden_states: np.ndarray) -> Dict[str, np.ndarray | float]:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    embedding = classical_mds(hidden_states, n_components=2)
    dist_hidden = pairwise_euclidean(hidden_states)
    dist_2d = pairwise_euclidean(embedding)
    numerator = np.sum((dist_hidden - dist_2d) ** 2)
    denominator = np.sum(dist_hidden ** 2)
    stress = np.sqrt(numerator / np.clip(denominator, 1e-12, None))
    return {"stress": float(stress), "embedding": embedding.astype(np.float32)}


def pca_variance_summary(hidden_states: np.ndarray) -> Dict[str, float | List[float]]:
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    n_components = min(hidden_states.shape[0], hidden_states.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(hidden_states)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    thresholds = {1: 0.0, 2: 0.0, 5: 0.0, 10: 0.0}
    for k in thresholds:
        thresholds[k] = float(np.sum(explained[: min(k, explained.shape[0])]))
    ninety = int(np.searchsorted(cumulative, 0.9) + 1)
    return {
        "top1": thresholds[1],
        "top2": thresholds[2],
        "top5": thresholds[5],
        "top10": thresholds[10],
        "n_for_90pct": ninety,
        "explained_variance": explained.tolist(),
    }

