"""Persistent-homology metrics and topology-convergence criteria."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional until runtime
    from ripser import ripser
except ImportError:  # pragma: no cover
    ripser = None


def _to_numpy(hidden_states) -> np.ndarray:
    if hasattr(hidden_states, "detach"):
        return hidden_states.detach().cpu().numpy()
    return np.asarray(hidden_states)


def _finite_lifetimes(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros((0,), dtype=np.float32)
    persistence = np.asarray(diagram[:, 1] - diagram[:, 0], dtype=np.float32)
    return np.sort(persistence[np.isfinite(persistence)])[::-1]


def _gap_ratio(lifetimes: np.ndarray, dominant_index: int = 0) -> float:
    if lifetimes.size <= dominant_index + 1:
        return 0.0
    numerator = float(lifetimes[dominant_index])
    denominator = float(max(lifetimes[dominant_index + 1], 1e-9))
    return numerator / denominator


def compute_betti_numbers(hidden_states, max_dim: int = 2, subsample: int = 300, seed: int = 42) -> dict[str, Any]:
    """Run ripser on CPU and compute coarse Betti summaries plus gap ratios."""
    if ripser is None:
        return {
            "betti_0": 0,
            "betti_1": 0,
            "diagrams": [],
            "persistence_gap_ratio_dim0": 0.0,
            "persistence_gap_ratio_dim1": 0.0,
            "persistence_gap_ratio_dim1_secondary": 0.0,
            "dim1_lifetimes": np.zeros((0,), dtype=np.float32),
            "available": False,
        }

    points = _to_numpy(hidden_states)
    if len(points) == 0:
        return {
            "betti_0": 0,
            "betti_1": 0,
            "diagrams": [],
            "persistence_gap_ratio_dim0": 0.0,
            "persistence_gap_ratio_dim1": 0.0,
            "persistence_gap_ratio_dim1_secondary": 0.0,
            "dim1_lifetimes": np.zeros((0,), dtype=np.float32),
            "available": True,
        }

    rng = np.random.default_rng(seed)
    if len(points) > subsample:
        keep = np.sort(rng.choice(len(points), size=subsample, replace=False))
        points = points[keep]

    result = ripser(points, maxdim=max_dim)
    diagrams = result["dgms"]
    dim0_lifetimes = _finite_lifetimes(diagrams[0]) if len(diagrams) > 0 else np.zeros((0,), dtype=np.float32)
    dim1_lifetimes = _finite_lifetimes(diagrams[1]) if len(diagrams) > 1 else np.zeros((0,), dtype=np.float32)

    def count_features(lifetimes: np.ndarray) -> int:
        if lifetimes.size == 0:
            return 0
        threshold = 0.1 * float(lifetimes[0])
        return int(np.sum(lifetimes > threshold))

    return {
        "betti_0": 1 if len(diagrams) > 0 and diagrams[0].shape[0] > 0 else 0,
        "betti_1": count_features(dim1_lifetimes),
        "diagrams": diagrams,
        "persistence_gap_ratio_dim0": _gap_ratio(dim0_lifetimes, dominant_index=0),
        "persistence_gap_ratio_dim1": _gap_ratio(dim1_lifetimes, dominant_index=0),
        "persistence_gap_ratio_dim1_secondary": _gap_ratio(dim1_lifetimes, dominant_index=1),
        "dim1_lifetimes": dim1_lifetimes,
        "available": True,
    }


def compute_betti_correct(betti_result: dict[str, Any], ground_truth: dict[str, int]) -> bool:
    """Require the correct coarse Betti counts plus non-trivial persistence separation."""
    if betti_result.get("betti_0") != ground_truth["betti_0"]:
        return False
    if betti_result.get("betti_1") != ground_truth["betti_1"]:
        return False
    if betti_result.get("persistence_gap_ratio_dim0", 0.0) <= 2.0:
        return False
    if ground_truth["betti_1"] == 0:
        return True
    if ground_truth["betti_1"] == 1:
        return bool(betti_result.get("persistence_gap_ratio_dim1", 0.0) > 2.0)
    if ground_truth["betti_1"] == 2:
        return bool(betti_result.get("persistence_gap_ratio_dim1_secondary", 0.0) > 2.0)
    return False


def compute_topology_convergence_step(topo_log: list[dict[str, Any]], ground_truth: dict[str, int], n_consecutive: int = 3):
    """First trial where the stricter topology criterion holds repeatedly."""
    streak = 0
    for entry in topo_log:
        if compute_betti_correct(entry, ground_truth):
            streak += 1
            if streak >= n_consecutive:
                return entry["trial"]
        else:
            streak = 0
    return None


def compute_geometry_convergence_step(geo_log: list[dict[str, Any]], threshold: float = 0.4, n_consecutive: int = 3):
    """First trial where geodesic sRSA crosses criterion repeatedly."""
    streak = 0
    for entry in geo_log:
        if float(entry.get("srsa_geodesic", -np.inf)) >= threshold:
            streak += 1
            if streak >= n_consecutive:
                return entry["trial"]
        else:
            streak = 0
    return None


def compute_convergence_gap(T_topology, T_geometry):
    """Positive values support topology-before-geometry."""
    if T_topology is None or T_geometry is None:
        return None
    return int(T_geometry) - int(T_topology)

