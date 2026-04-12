"""Aliasing controls and environment-level aliasing descriptors."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _grid_spacing(aliasing_level: str) -> int:
    aliasing_level = str(aliasing_level).lower()
    if aliasing_level == "low":
        return 2
    if aliasing_level == "medium":
        return 4
    if aliasing_level == "high":
        return 8
    if aliasing_level == "none":
        return 10_000
    raise ValueError(f"Unsupported aliasing level `{aliasing_level}`.")


def generate_tile_pattern(
    env_shape: tuple[int, int],
    aliasing_level: str,
    aliasing_type: str,
    seed: int = 42,
    colors: Iterable[str] = ("red", "blue", "yellow"),
) -> list[tuple[int, int, str]]:
    """Generate floor-tile anchors for discrete environments."""
    aliasing_type = str(aliasing_type).lower()
    if str(aliasing_level).lower() == "none":
        return []

    height, width = map(int, env_shape)
    colors = tuple(colors)
    spacing = _grid_spacing(aliasing_level)
    rng = np.random.default_rng(seed)

    if aliasing_type == "periodic":
        coords = [(row, col) for row in range(0, height, spacing) for col in range(0, width, spacing)]
    elif aliasing_type == "clustered":
        max_row = max(height // 2, 1)
        max_col = max(width // 2, 1)
        coords = [(row, col) for row in range(0, max_row, max(spacing // 2, 1)) for col in range(0, max_col, max(spacing // 2, 1))]
    elif aliasing_type == "sparse_random":
        n_tiles = max(1, math.ceil((height * width) / max(spacing ** 2, 1)))
        coords = list({(int(rng.integers(0, height)), int(rng.integers(0, width))) for _ in range(n_tiles * 3)})
        coords = coords[:n_tiles]
    else:
        raise ValueError(f"Unsupported aliasing type `{aliasing_type}`.")

    return [(col, row, colors[idx % len(colors)]) for idx, (row, col) in enumerate(coords)]


def compute_aliasing_score(env, n_samples: int = 500) -> float:
    """Estimate observation aliasing by comparing randomly separated samples."""
    rollout = env.sample_rollout(max(n_samples * 2, 32), seed=env.seed + 17)
    observations = np.asarray(rollout.observations, dtype=np.float32)
    positions = np.asarray(rollout.positions, dtype=np.float32)
    if len(observations) < 2:
        return 0.0

    norms = np.linalg.norm(observations, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    observations = observations / norms

    rng = np.random.default_rng(env.seed + 123)
    similarities: list[float] = []
    for _ in range(n_samples):
        idx_a, idx_b = rng.integers(0, len(observations), size=2)
        if np.linalg.norm(positions[idx_a] - positions[idx_b]) <= 3.0:
            continue
        similarities.append(float(np.dot(observations[idx_a], observations[idx_b])))
    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def compute_geo_euclidean_discrepancy(env, n_pairs: int = 2000) -> dict[str, float]:
    """Compare graph geodesic distances to Euclidean distances across random pairs."""
    rollout = env.sample_rollout(max(n_pairs * 2, 64), seed=env.seed + 29)
    positions = np.asarray(rollout.positions, dtype=np.float32)
    if len(positions) < 2:
        return {"mean": 1.0, "std": 0.0, "p95": 1.0}

    rng = np.random.default_rng(env.seed + 999)
    ratios: list[float] = []
    for _ in range(n_pairs):
        idx_a, idx_b = rng.integers(0, len(positions), size=2)
        if idx_a == idx_b:
            continue
        euclidean = float(np.linalg.norm(positions[idx_a] - positions[idx_b]))
        if euclidean <= 1e-6:
            continue
        geodesic = env.geodesic_distance(positions[idx_a], positions[idx_b])
        if not np.isfinite(geodesic):
            continue
        ratios.append(geodesic / euclidean)
    if not ratios:
        return {"mean": 1.0, "std": 0.0, "p95": 1.0}

    ratios_np = np.asarray(ratios, dtype=np.float32)
    return {
        "mean": float(np.mean(ratios_np)),
        "std": float(np.std(ratios_np)),
        "p95": float(np.percentile(ratios_np, 95)),
    }
