"""
File: project3_generalization/evaluation/topology.py

Description:
Persistent-homology helpers for characterizing the topology of neural state
clouds.

Role in system:
Used by Project 3 evaluation code to estimate Betti numbers and related
topological summaries from sampled hidden-state embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ripser import ripser
except ImportError:  # pragma: no cover
    ripser = None


def subsample_point_cloud(
    points: np.ndarray,
    max_points: int = 1_000,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Randomly downsample a point cloud before topology computation."""
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def persistent_homology(
    hidden_states: np.ndarray,
    *,
    maxdim: int = 2,
    max_points: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run `ripser` on a sampled hidden-state cloud and return the raw diagrams."""
    if ripser is None:
        raise ImportError("ripser is required for persistent homology metrics.")
    cloud = subsample_point_cloud(np.asarray(hidden_states, dtype=float), max_points=max_points, seed=seed)
    return ripser(cloud, maxdim=maxdim)


def betti_numbers_from_diagrams(
    diagrams: list[np.ndarray],
    *,
    persistence_fraction_threshold: float = 0.1,
) -> dict[str, Any]:
    """Convert persistence diagrams into coarse Betti-number estimates."""
    betti: dict[str, Any] = {}
    for dim, diagram in enumerate(diagrams):
        if diagram.size == 0:
            betti[f"b_{dim}"] = 0
            continue
        persistence = diagram[:, 1] - diagram[:, 0]
        finite = np.isfinite(persistence)
        finite_persistence = persistence[finite]
        if finite_persistence.size == 0:
            betti[f"b_{dim}"] = int(diagram.shape[0])
            continue
        threshold = persistence_fraction_threshold * float(finite_persistence.max())
        betti[f"b_{dim}"] = int(np.sum(finite_persistence > threshold))
    return betti


def compute_betti_numbers(
    hidden_states: np.ndarray,
    *,
    maxdim: int = 2,
    max_points: int = 1_000,
    seed: int = 0,
    persistence_fraction_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute persistent homology and package derived Betti-number summaries."""
    result = persistent_homology(
        hidden_states,
        maxdim=maxdim,
        max_points=max_points,
        seed=seed,
    )
    betti = betti_numbers_from_diagrams(
        result["dgms"],
        persistence_fraction_threshold=persistence_fraction_threshold,
    )
    betti["diagrams"] = result["dgms"]
    betti["point_cloud_size"] = min(len(hidden_states), max_points)
    return betti


__all__ = [
    "betti_numbers_from_diagrams",
    "compute_betti_numbers",
    "persistent_homology",
    "subsample_point_cloud",
]
